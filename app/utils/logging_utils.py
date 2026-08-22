import os
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# When Loguru starts, the default terminal handler ID is 0. This can only be replaced when WebUI reloads
# Basic terminal output, logger.remove() cannot be called to clear all handlers, otherwise the task is running
# The temporary sink used to collect WebUI logs will also be deleted.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()


def format_log_record(record):
    """
    Unified format terminal and WebUI log.

    Loguru The same record will be handed over to multiple sink. first one sink The absolute path may have been converted
    is a relative path to the project, so it is compatible with both absolute paths and ``./`` The formatted path that begins with .
    WebUI sink Colors are turned off, but the time, level, call location, and message content remain consistent with the terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
        record["file"].path = f"./{relative_path}"

    # Log messages sometimes contain the absolute path to the task file. Uniformly shorten to project relative path, you can
    # Prevent the WebUI and the terminal from displaying two sets of content due to different initialization entrances.
    record["message"] = record["message"].replace(PROJECT_ROOT, ".")
    return LOG_RECORD_FORMAT


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Safely replace process-level terminal logs handler, and keep the task-specific handler. 

    Streamlit Log initialization may be re-executed during code hot reload or cache invalidation. Just click on Recorded here
    of handler ID Precisely removes old terminal output, so it doesn't interrupt what the background task is writing WebUI
    log. The lock is used to protect multiple browser sessions when they are initialized simultaneously. ID renew.
    """
    global _terminal_handler_id

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # A test or external portal may have removed the handler. Go ahead and create a new terminal output,
                # There is no need to affect other log sinks that are still valid.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=level,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
