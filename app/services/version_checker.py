"""examine MoneyPrinterTurbo Is there a new official version available?"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import requests
from loguru import logger
from packaging.version import InvalidVersion, Version


from app.config import config

def _get_release_urls() -> tuple[str, str]:
    repo = getattr(config, "github_repo", "")
    if repo and "github.com/" in repo:
        clean_repo = repo.split("github.com/")[-1].strip().strip("/")
        return (
            f"https://api.github.com/repos/{clean_repo}/releases/latest",
            f"https://github.com/{clean_repo}/releases/latest",
        )
    return ("", "")

LATEST_RELEASE_API_URL: Final = ""
LATEST_RELEASE_PAGE_URL: Final = ""
# Update checking is only an auxiliary function, and network abnormalities cannot significantly slow down the local WebUI. Separate restrictions on connections and reads
# The timeout period not only allows GitHub to complete the response under normal network, but also avoids long waiting in offline environment.
RELEASE_CHECK_TIMEOUT: Final = (1.0, 2.0)
RELEASE_CHECK_HEADERS: Final = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "VideoCraftAI-Version-Checker",
}
UPDATE_CHECK_CACHE_TTL_SECONDS: Final = 12 * 60 * 60


def _parse_version(value: str) -> Version:
    """compatible GitHub Commonly used ``v1.2.3`` Label and convert to comparable versions."""
    normalized = str(value or "").strip()
    if normalized.lower().startswith("v"):
        normalized = normalized[1:]
    return Version(normalized)


def get_available_update(current_version: str) -> str | None:
    """
    Returns the latest official version higher than the current version; returned if there are no updates or the check fails ``None``. 
    """
    api_url, _ = _get_release_urls()
    if not api_url:
        return None

    try:
        installed_version = _parse_version(current_version)
    except InvalidVersion:
        logger.warning(
            f"skip update check because current version is invalid: {current_version!r}"
        )
        return None

    try:
        response = requests.get(
            api_url,
            headers=RELEASE_CHECK_HEADERS,
            timeout=RELEASE_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Update check failures are recoverable non-core exceptions. Retain exception types and information to facilitate locating agents,
        # DNS, GitHub throttling or response corruption issues while avoiding disturbing regular users in the WebUI.
        logger.debug(
            "GitHub release check failed: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None

    if not isinstance(payload, dict):
        logger.debug(
            "GitHub release check returned an invalid payload: "
            f"payload_type={type(payload).__name__}"
        )
        return None

    tag_name = payload.get("tag_name", "")
    try:
        latest_version = _parse_version(tag_name)
    except InvalidVersion:
        logger.warning(
            f"skip update notification because release tag is invalid: {tag_name!r}"
        )
        return None

    if latest_version <= installed_version:
        return None

    normalized_latest_version = str(latest_version)
    logger.info(
        "MoneyPrinterTurbo update available: "
        f"current={installed_version}, latest={normalized_latest_version}"
    )
    return normalized_latest_version


@dataclass(frozen=True)
class UpdateCheckSnapshot:
    """The real-time status of the background version check for WebUI Read without blocking."""

    complete: bool
    available_version: str | None = None


class AsyncUpdateChecker:
    """
    Perform version checking in a background thread and cache the latest results.

    Streamlit Page scripts will be executed from scratch after any control interaction. If accessed directly in the title area
    GitHub, the entire page will be blocked when first opened or when the cache expires. Here the network request is put into the daemon thread,
    The page only reads the current snapshot; after the check is completed, it is WebUI short term fragment Refresh the results once.

    Regardless of the result"Discover updates"still"no updates/Network failure"will be cached to avoid GitHub
    Inaccessible every time rerun Request all. The lock only protects the memory state and does not wrap the network request, so
    Does not block other sessions from reading check status.
    """

    def __init__(
        self,
        check: Callable[[str], str | None] = get_available_update,
        ttl_seconds: float = UPDATE_CHECK_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current_version: str | None = None
        self._available_version: str | None = None
        self._completed_at: float | None = None
        self._checking = False

    def poll(self, current_version: str) -> UpdateCheckSnapshot:
        """Return to check snapshot immediately; start a new check in the background when cache expires."""
        normalized_current_version = str(current_version or "").strip()
        now = self._clock()

        with self._lock:
            cache_is_fresh = (
                self._current_version == normalized_current_version
                and self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return UpdateCheckSnapshot(
                    complete=True,
                    available_version=self._available_version,
                )

            if (
                self._checking
                and self._current_version == normalized_current_version
            ):
                return UpdateCheckSnapshot(complete=False)

            # When the version changes or the cache expires, old results should not continue to be displayed. Clear the status first and then start
            # A new thread so that the caller gets an explicit snapshot of pending during the check.
            self._current_version = normalized_current_version
            self._available_version = None
            self._completed_at = None
            self._checking = True

            worker = threading.Thread(
                target=self._run_check,
                args=(normalized_current_version,),
                name="mpt-version-check",
                daemon=True,
            )
            worker.start()

        return UpdateCheckSnapshot(complete=False)

    def _run_check(self, current_version: str) -> None:
        try:
            available_version = self._check(current_version)
        except Exception:
            # get_available_update handles expected network and data exceptions. This is the background thread
            # Finally, to protect the boundary, the complete stack must be recorded to avoid permanent pending after unexpected exception and silent termination.
            logger.exception(
                "unexpected error while checking for a MoneyPrinterTurbo update"
            )
            available_version = None

        with self._lock:
            # In rare cases the version may change during operation. Old threads must not overwrite new versions of state.
            if self._current_version != current_version:
                return
            self._available_version = available_version
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_UPDATE_CHECKER = AsyncUpdateChecker()


def poll_available_update(current_version: str) -> UpdateCheckSnapshot:
    """Read global background checker status to avoid different Streamlit Session repeat request GitHub. """
    return _ASYNC_UPDATE_CHECKER.poll(current_version)
