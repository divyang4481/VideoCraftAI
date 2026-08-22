import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit allows larger uploaded files by default, but background music is typically only a few MB in size. Set clearly here
# The upper limit on the server side prevents the API or WebUI from completely writing extremely large files to disk and affecting video tasks in the same process.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
# MoviePy ultimately decodes background music via FFmpeg, so there's no need to artificially limit it to MP3. Only open here
# A mainstream and semantically clear audio extension to avoid mistakenly uploading video containers such as MP4 as background music.
# The tuple also serves as the single data source for the WebUI upload control, so there will be no inconsistency between the front and back ends when adding or deleting formats later.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """Indicates that the uploaded file does not meet the security or format requirements for background music."""


class BgmServiceError(RuntimeError):
    """express FFmpeg Or the file system is unavailable and other server-side execution failures."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Unifiedly determine whether the current task requires processing any background music.

    This rule is independent of the specific source: no source is selected, the volume is illegal, or the volume is not greater than 0 time, randomly,
    customized,Sonilo and future providers will have to skip file parsing, external generation and final mixing.
    put in general BGM You can avoid duplicating each provider in the service 0 Volume judgment.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Returns the persistent directory of the user's background music.

    The built-in songs belong to code resources and continue to be placed in resource/songs;The content uploaded by users belongs to runtime data.
    must be placed Docker Mounted storage , the container can be retained only after it is rebuilt, and it will not be polluted. Git work area.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Do your best to clean upload temporary files without overwriting the original exception being handled by the caller."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # Temporary files use reserved prefixes and will not enter the BGM list; failure to clean up should not cause "audio illegal"
        # Wait for the more accurate original exception to be covered, but the path and system errors must be left for operation and maintenance to locate.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extract audio file names that can be displayed across platforms, and reject illegal names and unsupported extensions."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(ord(character) < 32 for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows will recognize the first paragraph before the extension as the device name, such as CON.mp3 and LPT1.wav.
    # Cannot be created as a normal file. Even if the server ends up using UUIDs, early rejection of such names can
    # Ensure that API input behavior is consistent on different platforms.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Use only those currently configured for the project FFmpeg Verify that the file contains a fully decodable audio stream.

    Project allowed imageio-ffmpeg Provides portable FFmpeg, this installation method does not guarantee the simultaneous existence of
    FFprobe, so independent binary dependencies cannot be added.`-map 0:a:0` will fail when there is no audio stream,
    `-xerror` Will upgrade decoding errors to failures; complete decoding can also intercept encrypted files or random data accidentally
    Misjudgment of hitting audio frame header. The file can contain additional streams such as album art, but only the first audio stream is verified.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Verify that audio files on disk can be used by the project FFmpeg Full decoding.

    Upload preflight usually only requires 30 Second;Sonilo The generated soundtrack can be up to 6 minutes, so it is provided to the outside world
    Reuse entry with adjustable timeout. The service only depends on FFmpeg, does not require additional system installation FFprobe. 
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Write the upload stream to a temporary file in the same directory and return the safe file name, temporary path and number of bytes.

    WebUI The upload preflight and final persistence must use exactly the same chunked reads, size limits, and filenames
    Rules, otherwise there may be a state split where the interface displays available, but is rejected by the server after clicking to generate.
    Temporary files are deleted or replaced atomically by the caller after completing audio probing.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # Keeping the original extension allows FFmpeg to choose the correct one for formats such as AAC without container headers.
        # demuxer; temporary files are still placed in the target directory to ensure that the final os.replace operation is atomic.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("background music file exceeds the 30 MB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Streamlit also needs to use the same UploadedFile for browser listening; restoring the file pointer can
        # Prevent the player from reading empty content after verification or the final save.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Complete verification of uploaded audio but not persistence, used for WebUI Showing"Ready"Pre-inspection."""
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Save user background music in chunked, limited and atomic replacement methods.

    Usage scenarios include FastAPI UploadFile and Streamlit UploadedFile, both provide binary
    File interface. First write a temporary file in the same directory and verify it, and then pass os.replace Atomic placement can avoid
    Concurrent uploads or process interruptions leaving half of the audio file will also cause uploads with the same name to obtain different UUID storage key,
    Queued or running tasks therefore always reference the original immutable file.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def list_bgm_files() -> list[str]:
    """Lists available user-uploaded and built-in background music."""
    files_by_name: dict[str, str] = {}
    for directory in (utils.song_dir(), uploaded_bgm_dir(create=True)):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # Upload preflight and final save will briefly create files in the same directory. Although the temporary file has a legal
            # The audio extension has not yet been verified and cannot be pre-selected by the random BGM list.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # The enumeration results also need to be verified by the real path. Otherwise an attacker could place in an allowed directory
                # An audio symbolic link pointing to an external file and giving it to MoviePy with a random BGM path.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Parse in user upload directory and built-in song directory BGM, and reject paths outside the two whitelists.

    The file name hits the user directory first, while retaining `output000.mp3`, absolute whitelist path and
    `./resource/songs/output000.mp3` Wait for the old usage. Use for newly uploaded files UUID, under normal circumstances
    There will be no duplicate names with built-in songs or historical uploads.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
