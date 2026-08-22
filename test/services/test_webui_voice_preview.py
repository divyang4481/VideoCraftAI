import ast
import hashlib
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.models.schema import VideoParams
from app.services import task as tm
from app.services import voice
from app.services import webui_task
from app.utils import utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _load_duration_estimator():
    """,  Streamlit . """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_estimate_voiceover_duration_range"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace["_estimate_voiceover_duration_range"]


def _load_provider_signature(test_config):
    """ Provider , . """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_credential_signature",
            "_get_voice_preview_provider_signature",
        }
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"hashlib": hashlib, "config": test_config}
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace["_get_voice_preview_provider_signature"]


def _button_by_key(app, key):
    return next(
        button
        for button in app.button
        if str(getattr(button, "key", "")).startswith(key)
    )


def test_duration_estimator_is_local_and_respects_voice_rate():
    """, . """
    estimate = _load_duration_estimator()
    script = ". , . "

    normal_range = estimate(script, 1.0)
    fast_range = estimate(script, 2.0)

    assert normal_range is not None
    assert fast_range is not None
    assert normal_range[0] < normal_range[1]
    assert fast_range[0] < normal_range[0]
    assert estimate("", 1.0) is None
    assert estimate("AI tools can simplify repetitive work.", 1.0) is not None


def test_provider_signature_changes_when_api_key_changes():
    """ API Key , . """
    test_config = SimpleNamespace(
        app={"gemini_api_key": "old-gemini", "mimo_api_key": "old-mimo"},
        azure={"speech_region": "eastasia", "speech_key": "old-azure"},
        siliconflow={"api_key": "old-siliconflow"},
        elevenlabs={"api_key": "old-elevenlabs", "model_id": "eleven_v3"},
        chatterbox={
            "api_key": "old-chatterbox",
            "base_url": "http://127.0.0.1:4123/v1",
            "model_id": "chatterbox",
        },
    )
    provider_signature = _load_provider_signature(test_config)

    old_signature = provider_signature("elevenlabs")
    test_config.elevenlabs["api_key"] = "new-elevenlabs"
    new_signature = provider_signature("elevenlabs")

    assert old_signature != new_signature
    assert "old-elevenlabs" not in str(old_signature)
    assert "new-elevenlabs" not in str(new_signature)


def test_full_voiceover_preview_is_disabled_until_script_exists():
    """,  TTS. """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.run()

    full_preview = _button_by_key(
        app,
        "generate_full_voiceover_preview_button",
    )
    assert full_preview.disabled
    assert any("" in item.value for item in app.caption)


def test_script_shows_estimate_and_enables_full_voiceover_preview():
    """,  API . """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.session_state["video_script"] = (
            ". , . "
        )
        app.run()

    full_preview = _button_by_key(
        app,
        "generate_full_voiceover_preview_button",
    )
    assert not full_preview.disabled
    assert any(",  API" in item.value for item in app.caption)
    assert " API " in full_preview.help
    assert [str(item.value) for item in app.exception] == []


def test_short_preview_autoplays_only_after_explicit_click_and_reuses_cache():
    """;  rerun ,  TTS. """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts) as synthesize,
        patch.object(voice, "get_audio_duration", return_value=3.0),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.run()

        _button_by_key(app, "play_voice_button").click().run()
        assert len(app.get("audio")) == 1
        assert app.get("audio")[0].proto.autoplay

        app.run()
        assert len(app.get("audio")) == 1
        assert not app.get("audio")[0].proto.autoplay

        _button_by_key(app, "play_voice_button").click().run()

    synthesize.assert_called_once()
    assert app.get("audio")[0].proto.autoplay
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_uses_script_and_reuses_identical_cached_audio():
    """,  TTS. """
    script = ". "
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        # Although the file extension is mp3, the real TTS may return WAV; this minimum file header also
        # Verify that the WebUI recognizes player MIME by content rather than blindly by extension.
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts) as synthesize,
        patch.object(voice, "get_audio_duration", return_value=12.3),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.session_state["video_script"] = script
        app.run()

        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["text"] == script
    assert len(app.get("audio")) == 1
    assert not app.get("audio")[0].proto.autoplay
    assert any(": 12.3 " in item.value for item in app.caption)
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_reports_when_tts_returns_no_audio():
    """TTS , . """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", return_value=None),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.session_state["video_script"] = ". "
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    assert [item.value for item in app.error] == [
        ", . "
    ]
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_returns_immediately_when_runtime_config_is_busy():
    """, . """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(
            config,
            "try_runtime_config_lock",
            return_value=nullcontext(False),
        ),
        patch.object(voice, "tts") as synthesize,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.session_state["video_script"] = ". "
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    synthesize.assert_not_called()
    warning_messages = [item.value for item in app.warning]
    assert ", . " in warning_messages


def test_full_preview_warns_when_audio_duration_is_unavailable():
    """,  0.0 . """
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts),
        patch.object(voice, "get_audio_duration", return_value=0),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "zh"
        app.session_state["video_script"] = ". "
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    assert len(app.get("audio")) == 1
    warning_messages = [item.value for item in app.warning]
    assert ", , . " in warning_messages


def test_task_reuses_matching_full_preview_without_calling_tts():
    """, . """
    task_id = "reuse-full-voice-preview"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"preview audio")
    sub_maker = object()
    script = ". "
    params = VideoParams(
        video_subject="preview reuse",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.2,
        voice_volume=1.0,
    )
    preview = {
        "audio_file": str(audio_file),
        "duration": 8.2,
        "sub_maker": sub_maker,
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": params.voice_rate,
        "voice_volume": params.voice_volume,
    }

    try:
        with patch.object(tm.voice, "tts") as synthesize:
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result == (str(audio_file.resolve()), 9, sub_maker)
    synthesize.assert_not_called()


def test_task_regenerates_audio_when_preview_parameters_changed():
    """ TTS, . """
    task_id = "stale-full-voice-preview"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"stale preview audio")
    script = ". "
    params = VideoParams(
        video_subject="stale preview",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.5,
        voice_volume=1.0,
    )
    preview = {
        "audio_file": str(audio_file),
        "duration": 8.2,
        "sub_maker": object(),
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": 1.0,
        "voice_volume": params.voice_volume,
    }
    regenerated_sub_maker = object()

    try:
        with (
            patch.object(
                tm.voice,
                "tts",
                return_value=regenerated_sub_maker,
            ) as synthesize,
            patch.object(tm.voice, "get_audio_duration", return_value=6),
        ):
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result[1:] == (6, regenerated_sub_maker)
    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["voice_rate"] == 1.5


def test_non_default_volume_regenerates_audio_without_double_gain():
    """,  TTS . """
    task_id = "voice-volume-forwarding"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"preview with provider-side volume")
    script = ". "
    params = VideoParams(
        video_subject="voice volume",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.2,
        voice_volume=1.5,
    )
    sub_maker = object()
    preview = {
        "audio_file": str(audio_file),
        "duration": 5.0,
        "sub_maker": object(),
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": params.voice_rate,
        "voice_volume": params.voice_volume,
    }

    try:
        with (
            patch.object(tm.voice, "tts", return_value=sub_maker) as synthesize,
            patch.object(tm.voice, "get_audio_duration", return_value=5),
        ):
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result[1:] == (5, sub_maker)
    synthesize.assert_called_once()
    assert "voice_volume" not in synthesize.call_args.kwargs


def test_webui_worker_forwards_voice_preview_to_pipeline():
    """. """
    preview = {"audio_file": "audio.mp3", "duration": 5.0}
    with (
        patch.object(webui_task.tm, "start", return_value={"videos": []}) as start,
        patch.object(
            webui_task.config,
            "runtime_config_lock",
            return_value=nullcontext(),
        ),
    ):
        webui_task._run_generation(
            "preview-forwarding",
            VideoParams(video_subject="preview forwarding"),
            capture_logs=False,
            voice_preview=preview,
        )

    assert start.call_args.kwargs["voice_preview"] == preview
