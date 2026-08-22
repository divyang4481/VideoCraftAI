import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    def enqueue(self, task: Dict):
        task_with_serializable_params = task.copy()
        # task.copy() only copies the outermost dictionary; if you directly rewrite the nested kwargs, the caller will
        # The held VideoParams are synchronously replaced with dict. Subsequent logs or retries may still read the original task.
        # Therefore the kwargs are copied here separately to ensure there are no unintended side effects during the serialization process.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(task_kwargs["params"], VideoParams):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Convert a function object to its name
        task_with_serializable_params["func"] = task["func"].__name__
        self.redis_client.rpush(self.queue, json.dumps(task_with_serializable_params))

    def dequeue(self):
        # Loop instead of a single pop-up: a task may satisfy the current VideoParams verification rules when it is enqueued.
        # But the validation rules themselves were tightened between deployments (for example, the new ge=1 constraint was added). lpop is destructive
        # Operation, once it pops up, it cannot be put back into place; if you find that the verification fails when you rebuild VideoParams,
        # This task has been permanently removed from the queue and can no longer be pretended to be there. Instead of letting the exception go up from here
        # Throw and destroy the lock holder of this lost task. It is better to discard it in place and continue to try the queue.
        # The next item here is to maintain the agreement of "getting an available task or the queue is really empty".
        while True:
            task_json = self.redis_client.lpop(self.queue)
            if not task_json:
                return None

            task_info = json.loads(task_json)
            # Convert function name back to function object
            task_info["func"] = FUNC_MAP[task_info["func"]]

            if "params" in task_info["kwargs"] and isinstance(
                task_info["kwargs"]["params"], dict
            ):
                try:
                    task_info["kwargs"]["params"] = VideoParams(
                        **task_info["kwargs"]["params"]
                    )
                except ValidationError as e:
                    logger.error(
                        "dropping queued task with params that fail current "
                        f"VideoParams validation (queued under an older, more "
                        f"permissive schema, or corrupted): {e}"
                    )
                    # The task status record is created before being queued, and the default is processing; if only
                    # Discard this queue item without touching the status record. The API/WebUI will always display that the task is in progress.
                    # Run, never turn into failure. Use patch_task instead of update_task,
                    # This way if the user has deleted the task, we will not create it back again.
                    task_id = task_info["kwargs"].get("task_id")
                    if task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"discarded stale queued task: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
