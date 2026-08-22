import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Keep dict The usage method remains unchanged, and at the same time, the runtime configuration write operation is subject to the same lock."""

    def __setitem__(self, key, value):
        # Streamlit will rewrite the current control value back to the configuration every time the entire page is rerun. video task holding
        # When runtime_config_lock, if the value does not change, this writing has no side effects, and
        # The refreshed page should not be stuck in the middle of the form. Writes that actually change the configuration still go into the lower lock,
        # Therefore, you cannot switch providers, keys, or other global settings in the middle of a video being generated.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` also does not change the configuration when key does not exist. WebUI usage
        # This way of writing expresses "adopting the default policy", which must be allowed to complete directly when refreshing.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # Like __setitem__, setdefault for an existing key is a read-only operation. Return early
        # This allows page refreshes that only read the default configuration to be unaffected by long task configuration locks.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Generate keys to be updated for in-process fixed configuration partitions."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    non-blocking update WebUI runtime configuration.

    Video generation will hold ``runtime_config_lock``, ensuring that the same task will not be switched mid-execution
    Provider, key or voice configuration.Streamlit You cannot wait for this long task lock when the control changes.
    Otherwise the browser will appear to freeze the page. Update immediately when the lock is idle; only retain each configuration item when the lock is busy
    The latest value is applied uniformly when the current task releases the lock.

    return True Indicates that the value has taken effect,False Indicates that it has entered the queue to be updated.
    """
    # All updates are put into the same queue before trying to acquire the configuration lock. In this way, multiple pages can modify the same page at the same time.
    # When configuring items, the order of writing to the queue is the final order, and there will be no earlier threads after acquiring the lock.
    # Values ​​already queued by newer threads were mistakenly deleted.
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # The caller will usually request a save at the end of this Streamlit rerun, but cannot rely on this step
        # Must be implemented. For example, if the page is abnormal in the middle or the update happens to occur when the task exits the save phase, it still needs to be
        # There is a background refresh thread to ensure that the queued value finally takes effect.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    non-blocking delete WebUI Configuration items.

    "Use default value"Need to actually remove the configuration item, rather than write an empty string. Video task occupation configuration
    When locked, the deletion intent overwrites previously queued updates to the same configuration item and is executed after the task ends.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Applied while holding configuration write lock WebUI The latest temporary configuration value."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # The pending update lock continues to be held while applying the configuration. The thread that reads the "current value + value to be updated" snapshot is thus
        # Only the complete status before or after application can be seen, and only half-updated configuration collections will not be read.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Returns a valid snapshot of the configuration partition, merging any that have not yet been applied WebUI renew.

    The global configuration cannot be rewritten while the video task is locked, but the user can still prepare the next piece of content.LLM ask
    After using this snapshot, the newly selected Provider, model and key will participate in the new request, while
    Does not change the video task being performed.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Applies and saves all currently pending configurations while holding a configuration write lock."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # The configuration in the memory has been successfully applied. If the save fails, only the mark to be saved is retained. Video tasks should not
        # The modification failed because the configuration file is temporarily unwritable; the next page interaction will trigger saving again.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Wait for long tasks to release configuration locks and reliably clear configuration updates accumulated during the period."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """It is guaranteed that there is at most one background thread waiting to refresh the configuration at the same time."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    threading.Thread(
        target=_run_deferred_config_flush,
        name="mpt-config-flush",
        daemon=True,
    ).start()


def try_save_config():
    """
    non-blocking save WebUI Configuration, when the lock is busy, it will be saved after the current long task ends.

    ordinary API, CLI and maintenance scripts can still be called ``save_config`` Get the original blocking write semantics;
    only Streamlit rerun Use this function to prevent the page from being unresponsive for a long time while waiting for video tasks.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Block other operations during a complete operation that relies on global configuration WebUI Session override configuration.

    The current project binds the local loopback address by default, and the configuration is still a single-user global configuration. This lightweight lock mainly
    Protect long operations such as generation and audition to prevent another tab from switching mid-operation. Provider or key.
    """
    with _config_save_lock:
        # If the background refresh thread has not yet been scheduled when the previous short operation releases the lock, the new task must be read
        # The queue is applied before global configurations such as providers and keys. You cannot continue to use the old configuration to execute the entire pipeline.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Attempts to obtain a runtime configuration lock and returns immediately whether successful.

    WebUI Audition is a short operation triggered by the user and should not wait for several minutes while the background video task is locked.
    The caller can prompt the user to try again later when the lock is not acquired; after successfully acquiring the lock, the listening period can still be guaranteed.
    Provider, keys, and model configurations will not be modified by other sessions.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Determine whether the current process is running in the container.

    This judgment is mainly used for Ollama Default address selection:
    - When running normally on this machine,`localhost` Points to the user's machine itself;
    - Docker inside the container,`localhost` Point to the container itself and access the host Ollama
      Usually it is necessary to use `host.docker.internal`. 

    Don't just judge `/proc/1/cgroup` Does it exist because ordinary Linux There will also be this file.
    This only returns when an explicit container tag is detected True, to avoid accidental injury to non- Docker Linux user.
    Parameters are reserved as injectable paths to facilitate unit testing to cover different operating environments.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # The Gateway in /proc/net/route is hexadecimal little endian, for example, 010011AC means
    # 172.17.0.1. It is parsed separately here in order to use it when native Linux Docker does not have
    # host.docker.internal DNS record, it can also try to access the host on the container's default gateway.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    read Linux The default gateway in the container IP. 

    Docker Desktop usually provided `host.docker.internal`, but native Linux Docker
    This is not necessarily provided by default DNS name. The default gateway can usually be used to access host services.
    Dirty address; if the user's Ollama Only listen 127.0.0.1, the user still needs to let
    Ollama Monitor the host network card or configure it manually `ollama_base_url`. 
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 3:
            continue

        destination = fields[1]
        gateway = fields[2]
        if destination != "00000000" or gateway == "00000000":
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    return Ollama the default OpenAI-compatible base_url. 

    User explicit configuration `ollama_base_url` I won't go here; I only deal with"When not configured
    best default". The container points to the host by default, and the normal local machine runs by default. localhost. 
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    load TOML, and compatible with Windows Editor may write duplicates UTF-8 BOM. 

    ``utf-8-sig`` Only the one at the beginning of the file will be removed BOM. part Windows editor or
    The decompression and saving process may be written again BOM, causing the second invisible character to enter TOML
    The parser reports an error on the first line. Here only a read-only normalization is done after the standard parsing fails,
    Do not write back the original file to avoid accidentally overwriting the information already filled in by the user. API Key. 
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/MoneyPrinterTurbo/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


def save_config():
    """
    Atomic saving of runtime configuration.

    Streamlit Different sessions may trigger configuration saves at similar times. direct coverage config.toml hour,
    Another thread may read only partially written TOML content. In-process reentrant lock serialization is used here
    Save, and first write to the temporary file in the same directory, and then pass os.replace Atomic replacement of target files.

    Docker Desktop single file bind mount will config.toml itself as a mount point,
    Linux Kernel does not allow pass rename/replace Replaces the mount point, so it returns EBUSY. 
    In this scenario, the file can only be overwritten in place within the lock; other exceptions are still thrown to avoid covering permissions, disk
    Or the path is wrong.

    This still retains the project's existing single-user global configuration semantics without introducing an additional complex multi-user configuration system;
    Mainly used to avoid multiple tabs or quickly rerun The configuration file is damaged.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["ui"] = dict(ui)
        serialized_config = toml.dumps(config_to_save)

        # Save will be called at the end of a complete rerun of WebUI. Return directly when the content has not changed to avoid each time
        # Clicking on a normal control will cause a disk write and fsync.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "VideoCraft AI")
project_description = _cfg.get(
    "project_description",
    "✨ VideoCraft AI - Advanced AI Automated Video Production Studio",
)
github_repo = _cfg.get("github_repo", "")
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
