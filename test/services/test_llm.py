import json
import os
import sys
import tempfile
import tomllib
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    LLM_PROVIDERS,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import VideoScriptRequest, VideoSocialMetadataRequest
from app.services import llm

RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestScriptPromptOptions(unittest.TestCase):
    def test_normalize_text_response_preserves_internal_newlines(self):
        """
        , . 
        , . 
        """
        result = llm._normalize_text_response(
            "\n  \n\n\n  \n",
            "openai",
        )

        self.assertEqual(result, "\n\n\n")

    def test_normalize_text_response_removes_think_blocks(self):
        """
        reasoning  `<think>...</think>`. 
        , . 
        """
        result = llm._normalize_text_response(
            "<think>\nI should reason here.\n</think>\n",
            "minimax",
        )

        self.assertEqual(result, "")

    def test_normalize_text_response_rejects_think_only_response(self):
        """
        , , . 
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning</think>", "minimax")

    def test_normalize_text_response_removes_unclosed_think_block(self):
        """
         `<think>`. 
        ; , . 
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning", "minimax")

    def test_build_script_prompt_appends_advanced_requirements(self):
        """
        , . 
        , . 
        """
        prompt = llm.build_script_prompt(
            video_subject="",
            language="zh-CN",
            paragraph_number=3,
            video_script_prompt=", ",
        )

        self.assertIn("# Role: Video Script Generator", prompt)
        self.assertIn("- video subject: ", prompt)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("- language: zh-CN", prompt)
        self.assertIn("# Additional User Requirements:", prompt)
        self.assertIn(", ", prompt)

    def test_custom_system_prompt_keeps_runtime_context(self):
        """
         system prompt , , , 
        , . 
        """
        prompt = llm.build_script_prompt(
            video_subject="",
            language="en",
            paragraph_number=2,
            custom_system_prompt="Only write cinematic narration.",
        )

        self.assertNotIn("# Role: Video Script Generator", prompt)
        self.assertIn("Only write cinematic narration.", prompt)
        self.assertIn("- video subject: ", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("- language: en", prompt)

    def test_generate_script_sends_custom_prompt_to_llm(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return ". \n\n. "

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="",
                language="zh-CN",
                paragraph_number=2,
                video_script_prompt="",
            )

        self.assertEqual(result, ". \n\n. ")
        self.assertIn("- number of paragraphs: 2", captured["prompt"])
        self.assertIn("", captured["prompt"])

    def test_generate_script_reuses_submitted_config_snapshot(self):
        """WebUI , . """
        captured = {}
        app_config = {
            "llm_provider": "openai",
            "openai_api_key": "snapshot-key",
            "openai_model_name": "snapshot-model",
        }

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            captured["app_config"] = app_config
            return "Snapshot response"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="Snapshot test",
                app_config=app_config,
            )

        self.assertEqual(result, "Snapshot response")
        self.assertIs(captured["app_config"], app_config)
        self.assertEqual(captured["app_config"]["openai_api_key"], "snapshot-key")

    def test_generate_terms_can_request_script_ordered_keywords(self):
        """
         LLM . , 
        "" prompt, 
        , . 
        """
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return '["opening city", "middle office", "final sunset"]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])
        self.assertIn("chronological stock-video search terms", captured["prompt"])
        self.assertIn("same order as the script narration", captured["prompt"])

    def test_generate_terms_returns_empty_list_on_provider_error(self):
        """
        Provider  generate_terms  List[str] . 

         ``Error: ...``  Python ; , 
        , . 
        """
        with patch.object(
            llm,
            "_generate_response",
            return_value="Error: invalid API key",
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="A short startup story.",
            )

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)

    def test_video_script_request_rejects_invalid_advanced_options(self):
        """
        API  prompt ,  WebUI
        , . 
        """
        with self.assertRaises(ValidationError):
            VideoScriptRequest(video_subject="", paragraph_number=0)

        with self.assertRaises(ValidationError):
            VideoScriptRequest(
                video_subject="",
                video_script_prompt="x" * (llm.MAX_SCRIPT_PROMPT_LENGTH + 1),
            )


class TestLLMConnection(unittest.TestCase):
    def test_connection_sends_one_minimal_request(self):
        """, . """
        with (
            patch.object(llm, "_generate_response", return_value="OK") as generate,
            patch.object(llm, "perf_counter", side_effect=[10.0, 10.25]),
        ):
            result = llm.test_connection()

        generate.assert_called_once_with(prompt="Reply with exactly: OK")
        self.assertEqual(result, (True, "", 0.25))

    def test_connection_returns_provider_error(self):
        """Provider , . """
        with (
            patch.object(
                llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(llm, "perf_counter", side_effect=[20.0, 20.5]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "invalid API key", 0.5))

    def test_connection_rejects_empty_response(self):
        """, . """
        with (
            patch.object(llm, "_generate_response", return_value=""),
            patch.object(llm, "perf_counter", side_effect=[30.0, 31.0]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "LLM returned an empty response", 1.0))


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_current_default_model_names(self):
        """WebUI , . """
        self.assertEqual(get_llm_provider("openai").default_model, "gpt-5.5")
        anthropic = get_llm_provider("anthropic")
        self.assertEqual(anthropic.default_model, "claude-sonnet-5")
        self.assertEqual(anthropic.default_base_url, "https://api.anthropic.com/v1/")
        self.assertEqual(anthropic.adapter, "openai_compatible")
        self.assertTrue(anthropic.requires_api_key)
        self.assertEqual(get_llm_provider("aimlapi").default_model, "openai/gpt-5-5")
        self.assertEqual(get_llm_provider("deepseek").default_model, "deepseek-v4-pro")
        self.assertEqual(
            get_llm_provider("modelscope").default_model, "ZhipuAI/GLM-5.2"
        )
        self.assertEqual(
            get_llm_provider("gemini").default_model, "gemini-3.1-pro-preview"
        )
        pollinations = get_llm_provider("pollinations")
        self.assertEqual(pollinations.default_model, "openai-fast")
        self.assertEqual(
            pollinations.default_base_url,
            "https://gen.pollinations.ai/v1",
        )
        self.assertTrue(pollinations.requires_api_key)
        self.assertEqual(pollinations.adapter, "openai_compatible")

    def test_provider_defaults_are_not_persisted_as_user_overrides(self):
        """, . """
        self.assertEqual(
            normalize_provider_override("gpt-5.5", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("  gpt-5.5  ", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("gpt-5.6-custom", "gpt-5.5"),
            "gpt-5.6-custom",
        )

    def test_provider_registry_has_unique_stable_ids(self):
        """Registry  Provider , ID . """
        provider_ids = [provider.provider_id for provider in LLM_PROVIDER_REGISTRY]

        self.assertEqual(len(provider_ids), len(set(provider_ids)))
        self.assertEqual(len(provider_ids), len(LLM_PROVIDERS))
        self.assertIn(DEFAULT_LLM_PROVIDER_ID, LLM_PROVIDERS)

    def test_provider_registry_preserves_product_group_order(self):
        """, , , . """
        self.assertEqual(
            [provider.provider_id for provider in LLM_PROVIDER_REGISTRY],
            [
                "moonshot",
                "openai",
                "anthropic",
                "gemini",
                "deepseek",
                "qwen",
                "azure",
                "volcengine",
                "grok",
                "minimax",
                "mimo",
                "shengsuanyun",
                "cloudflare",
                "modelscope",
                "aihubmix",
                "aimlapi",
                "evolink",
                "ollama",
                "oneapi",
                "litellm",
                "groq",
                "pollinations",
            ],
        )
        self.assertEqual(
            get_llm_provider("gemini").default_label,
            "Google Gemini",
        )
        self.assertEqual(
            get_llm_provider("azure").default_label,
            "Microsoft Azure OpenAI",
        )
        shengsuanyun = get_llm_provider("shengsuanyun")
        self.assertEqual(
            shengsuanyun.api_key_url,
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        )
        self.assertEqual(
            shengsuanyun.default_model,
            "deepseek/deepseek-v4-flash",
        )

    def test_provider_registry_uses_conventional_locale_and_config_keys(self):
        """ WebUI  Provider . """
        for provider in LLM_PROVIDER_REGISTRY:
            self.assertEqual(
                provider.label_key,
                f"llm_provider_label.{provider.provider_id}",
            )
            self.assertEqual(
                provider.tips_key,
                f"llm_provider_tips.{provider.provider_id}",
            )
            self.assertEqual(
                provider.config_key("api_key"),
                f"{provider.provider_id}_api_key",
            )

    def test_registry_replaces_deprecated_provider_models(self):
        """, . """
        cloudflare = get_llm_provider("cloudflare")
        gemini = get_llm_provider("gemini")

        self.assertEqual(
            cloudflare.resolve_model_name("@cf/meta/llama-3.1-8b-instruct"),
            "openai/gpt-4.1-mini",
        )
        self.assertEqual(
            gemini.resolve_model_name("gemini-pro"),
            "gemini-3.1-pro-preview",
        )
        self.assertEqual(
            cloudflare.resolve_model_name("anthropic/claude-sonnet-4-5"),
            "anthropic/claude-sonnet-4-5",
        )

        pollinations = get_llm_provider("pollinations")
        self.assertEqual(
            pollinations.resolve_model_name("default"),
            "openai-fast",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://text.pollinations.ai/openai"),
            "https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://example.com/v1"),
            "https://example.com/v1",
        )

    def test_provider_tip_templates_accept_registry_defaults(self):
        """ Provider  Registry . """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for locale_file in i18n_dir.glob("*.json"):
            translations = json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations.get(provider.tips_key, "")
                if not tips:
                    continue
                default_endpoint = provider.default_service_endpoint
                rendered = tips.format(
                    api_key_url=provider.effective_api_key_url(),
                    default_model=provider.default_model,
                    default_base_url=provider.effective_default_base_url,
                    model_docs_url=(
                        default_endpoint.model_docs_url
                        if default_endpoint
                        else ""
                    ),
                    docker_hint="",
                    **{
                        f"default_{field.config_suffix}": field.default_value
                        for field in provider.extra_fields
                    },
                )
                self.assertNotIn("{default_model}", rendered)
                self.assertNotIn("{default_base_url}", rendered)

    def test_primary_provider_tips_use_consistent_structure(self):
        """ API Key, Base URL . """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for language in ("zh", "en"):
            translations = json.loads(
                (i18n_dir / f"{language}.json").read_text(encoding="utf-8")
            )["Translation"]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations[provider.tips_key]
                self.assertTrue(tips.startswith("##### "), provider.provider_id)
                self.assertIn("**API Key**", tips, provider.provider_id)
                self.assertIn("**Base Url**", tips, provider.provider_id)
                self.assertIn("**Model Name**", tips, provider.provider_id)

        zh_kimi_tips = json.loads((i18n_dir / "zh.json").read_text(encoding="utf-8"))[
            "Translation"
        ]["llm_provider_tips.moonshot"]
        self.assertIn(": ", zh_kimi_tips)
        self.assertIn("", zh_kimi_tips)

    def test_required_api_key_providers_have_clickable_entry_points(self):
        """ Provider ,  WebUI . """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        locale_translations = {
            locale_file.stem: json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for locale_file in i18n_dir.glob("*.json")
        }

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.requires_api_key:
                api_key_url = provider.effective_api_key_url()
                self.assertTrue(api_key_url, provider.provider_id)
                self.assertTrue(
                    api_key_url.startswith("https://"),
                    provider.provider_id,
                )
                for language, translations in locale_translations.items():
                    tips_template = translations.get(provider.tips_key, "")
                    if not tips_template:
                        continue
                    default_endpoint = provider.default_service_endpoint
                    tips = tips_template.format(
                        api_key_url=api_key_url,
                        default_model=provider.default_model,
                        default_base_url=provider.effective_default_base_url,
                        model_docs_url=(
                            default_endpoint.model_docs_url
                            if default_endpoint
                            else ""
                        ),
                        docker_hint="",
                        **{
                            f"default_{field.config_suffix}": field.default_value
                            for field in provider.extra_fields
                        },
                    )
                    api_key_line = next(
                        line for line in tips.splitlines() if "**API Key**" in line
                    )
                    self.assertIn("](", api_key_line, provider.provider_id)
                    self.assertIn(
                        f"]({api_key_url})",
                        api_key_line,
                        f"{language}: {provider.provider_id}",
                    )

    def test_service_endpoint_registry_references_valid_stable_ids(self):
        """ ID , . """
        for provider in LLM_PROVIDER_REGISTRY:
            endpoint_ids = [
                endpoint.endpoint_id for endpoint in provider.service_endpoints
            ]
            self.assertEqual(
                len(endpoint_ids),
                len(set(endpoint_ids)),
                provider.provider_id,
            )
            if not endpoint_ids:
                self.assertFalse(provider.default_service_endpoint_id)
                self.assertFalse(provider.international_service_endpoint_id)
                continue

            self.assertIn(provider.default_service_endpoint_id, endpoint_ids)
            if provider.international_service_endpoint_id:
                self.assertIn(provider.international_service_endpoint_id, endpoint_ids)

    def test_kimi_service_endpoint_selection_preserves_existing_configs(self):
        """ Kimi . """
        provider = get_llm_provider("moonshot")

        china = provider.select_service_endpoint(
            "",
            has_api_key=True,
            prefer_international=True,
        )
        global_endpoint = provider.select_service_endpoint(
            "https://api.moonshot.ai/v1/",
            has_api_key=True,
            prefer_international=False,
        )

        self.assertEqual(china.endpoint_id, "china")
        self.assertEqual(global_endpoint.endpoint_id, "global")
        self.assertIsNone(
            provider.select_service_endpoint(
                "https://gateway.example.com/v1",
                has_api_key=True,
                prefer_international=True,
            )
        )

    def test_kimi_fresh_config_uses_interface_region(self):
        """,  WebUI . """
        provider = get_llm_provider("moonshot")

        china = provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=False,
        )
        global_endpoint = provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=True,
        )

        self.assertEqual(china.base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(global_endpoint.base_url, "https://api.moonshot.ai/v1")
        self.assertIn("platform.kimi.ai", global_endpoint.api_key_url)

    def test_kimi_endpoint_selection_does_not_depend_on_marketing_url(self):
        """. """
        provider = get_llm_provider("moonshot")
        global_endpoint = replace(
            provider.international_service_endpoint,
            api_key_url="https://platform.kimi.ai/?new-tracking=1",
        )
        updated_provider = replace(
            provider,
            service_endpoints=tuple(
                global_endpoint if endpoint.endpoint_id == "global" else endpoint
                for endpoint in provider.service_endpoints
            ),
        )

        selected = updated_provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=True,
        )

        self.assertEqual(selected.endpoint_id, "global")
        self.assertEqual(selected.api_key_url, global_endpoint.api_key_url)

    def test_example_config_does_not_duplicate_registry_defaults(self):
        """,  Registry . """
        config_path = Path(__file__).parent.parent.parent / "config.example.toml"
        app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.default_model:
                self.assertEqual(
                    app_config.get(provider.config_key("model_name"), ""),
                    "",
                    provider.provider_id,
                )
            if provider.effective_default_base_url:
                self.assertEqual(
                    app_config.get(provider.config_key("base_url"), ""),
                    "",
                    provider.provider_id,
                )
            for field in provider.extra_fields:
                if field.default_value:
                    self.assertEqual(
                        app_config.get(provider.config_key(field.config_suffix), ""),
                        "",
                        provider.provider_id,
                    )

    def test_removed_ernie_provider_is_unsupported(self):
        """ ERNIE , ,  OAuth . """
        config.app["llm_provider"] = "ernie"

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("unsupported llm provider", result)

    def test_pollinations_requires_api_key_before_request(self):
        """ API ,  Key . """
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "",
                "pollinations_base_url": "",
                "pollinations_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("api_key is not set", result)

    def test_pollinations_uses_unified_openai_compatible_api(self):
        """,  Chat Completions API . """
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "pollinations-test-key",
                "pollinations_base_url": "https://text.pollinations.ai/openai/",
                "pollinations_model_name": "default",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\npollinations")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai-fast",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\npollinations")

    def test_anthropic_uses_openai_compatible_chat_completions(self):
        """Claude  Anthropic  OpenAI , . """
        config.app.update(
            {
                "llm_provider": "anthropic",
                "anthropic_api_key": "anthropic-test-key",
                "anthropic_base_url": "",
                "anthropic_model_name": "",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nclaude")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="anthropic-test-key",
            base_url="https://api.anthropic.com/v1/",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nclaude")

    def test_gemini_uses_google_genai_client(self):
        """Gemini  SDK  Client . """
        config.app.update(
            {
                "llm_provider": "gemini",
                "gemini_api_key": "gemini-test-key",
                "gemini_base_url": "",
                "gemini_model_name": "gemini-test-model",
            }
        )
        captured = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text="hello\ngemini")

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        with patch("google.genai.Client", FakeClient):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\ngemini")
        self.assertEqual(
            captured["client_kwargs"],
            {"api_key": "gemini-test-key", "http_options": None},
        )
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(captured["contents"], "Say hello")
        self.assertEqual(captured["config"].max_output_tokens, 2048)
        self.assertTrue(captured["closed"])

    def test_cloudflare_requires_account_id_before_request(self):
        """Cloudflare  Account ID , . """
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "test-token",
                "cloudflare_account_id": "",
                "cloudflare_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("account_id is not set", result)

    def test_cloudflare_uses_ai_gateway_openai_endpoint(self):
        """Cloudflare Provider  AI Gateway,  Workers AI . """
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "cloudflare-token",
                "cloudflare_account_id": "account-123",
                "cloudflare_gateway_id": "",
                "cloudflare_model_name": "",
            }
        )

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="gateway\nresponse")
                )
            ]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return fake_response

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="cloudflare-token",
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
            ),
            default_headers={"cf-aig-gateway-id": "default"},
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "gateway\nresponse")

    def _use_litellm_provider(self, model_name="openai/gpt-4o-mini"):
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = model_name

    def test_litellm_provider_returns_normalized_text(self):
        """
         LiteLLM provider  API key. 

         fake module  `sys.modules`,  import 
        `litellm.completion()`,  `_generate_response()` 
        litellm . 
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                kwargs["messages"], [{"role": "user", "content": "Say hello"}]
            )
            self.assertTrue(kwargs["drop_params"])
            message = types.SimpleNamespace(content="hello\nworld")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\nworld")

    def test_litellm_provider_uses_registry_default_model(self):
        self._use_litellm_provider(model_name="")

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            message = types.SimpleNamespace(content="default model")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertEqual(result, "default model")

    def test_litellm_provider_handles_empty_response(self):
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[])
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty response", result)

    def test_litellm_provider_handles_empty_message(self):
        """
         OpenAI-compatible 
        HTTP 200,  `choices[0].message`  None. 
        ,  AttributeError. 
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=None)]
            )
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty message", result)

    def test_sanitize_error_message_redacts_url_credentials_and_query_tokens(self):
        message = (
            "request failed for "
            "https://myuser:mypassword@proxy.example.com/v1/chat"
            "?api_key=secret-key&token=secret-token&safe=value"
        )

        result = llm._sanitize_error_message(message)

        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("api_key=***", result)
        self.assertIn("token=***", result)
        self.assertIn("safe=value", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-key", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_error_redacts_embedded_base_url_credentials(self):
        """
         OpenAI-compatible base_url  user:pass. 
        SDK  URL ,  WebUI/API 
        `Error:` . 
        """
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-key"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"
        config.app["groq_base_url"] = (
            "https://myuser:mypassword@proxy.example.com/openai/v1"
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise RuntimeError(
                    "connection failed: "
                    "https://myuser:mypassword@proxy.example.com/openai/v1"
                    "?access_token=secret-token"
                )

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        with patch.object(llm, "OpenAI", return_value=fake_client):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("access_token=***", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = ""
        config.app["openai_base_url"] = "https://api.openai.com/v1"
        config.app["openai_model_name"] = "gpt-4o-mini"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def _use_qwen_provider(self):
        config.app["llm_provider"] = "qwen"
        config.app["qwen_api_key"] = "qwen-key"
        config.app["qwen_model_name"] = "qwen-max"

    def _patch_dashscope_generation(self, response):
        class FakeGenerationResponse(dict):
            pass

        fake_response = FakeGenerationResponse(response)
        fake_response.status_code = response.get("status_code", 200)
        fake_dashscope = types.SimpleNamespace(
            api_key="",
            Generation=types.SimpleNamespace(call=lambda **kwargs: fake_response),
        )
        fake_dashscope_response = types.SimpleNamespace(
            GenerationResponse=FakeGenerationResponse
        )

        return patch.dict(
            sys.modules,
            {
                "dashscope": fake_dashscope,
                "dashscope.api_entities": types.SimpleNamespace(),
                "dashscope.api_entities.dashscope_response": fake_dashscope_response,
            },
        )

    def test_qwen_provider_reads_chat_choices_content(self):
        """
        DashScope chat  `output.choices[0].message.content`. 
         issue # This covers the `output.text is None` scenario reported in issue #966 to avoid triggering again
        `'NoneType' object has no attribute 'replace'`. 
        """
        self._use_qwen_provider()
        response = {
            "output": {
                "text": None,
                "choices": [{"message": {"content": "\n"}}],
            }
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "\n")

    def test_qwen_provider_falls_back_to_output_text(self):
        """ DashScope completion . """
        self._use_qwen_provider()
        response = {"output": {"text": "\n"}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "\n")

    def test_qwen_provider_reports_empty_text(self):
        """Qwen ,  AttributeError. """
        self._use_qwen_provider()
        response = {
            "output": {"text": None, "choices": [{"message": {"content": None}}]}
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty text content", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_provider_reports_empty_choices(self):
        """Qwen chat  choices . """
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": []}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty choices", result)
        self.assertNotIn("NoneType", result)

    def test_aihubmix_provider_uses_openai_compatible_client(self):
        """
        AIHubMix  OpenAI-compatible .  fake OpenAI client
         Provider  Registry , 
         API Key . 
        """
        config.app["llm_provider"] = "aihubmix"
        config.app["aihubmix_api_key"] = "aihubmix-key"
        config.app["aihubmix_base_url"] = ""
        config.app["aihubmix_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naihubmix")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aihubmix-key",
            base_url="https://aihubmix.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naihubmix")

    def test_aimlapi_provider_uses_openai_compatible_client(self):
        config.app["llm_provider"] = "aimlapi"
        config.app["aimlapi_api_key"] = "aimlapi-key"
        config.app["aimlapi_base_url"] = ""
        config.app["aimlapi_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naimlapi")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aimlapi-key",
            base_url="https://api.aimlapi.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-5-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naimlapi")

    def test_evolink_provider_uses_openai_compatible_client(self):
        """
        EvoLink exposes OpenAI-compatible Chat Completions at direct.evolink.ai.
        The provider should keep its own default endpoint and model instead of
        requiring users to overload the generic OpenAI settings.
        """
        config.app["llm_provider"] = "evolink"
        config.app["evolink_api_key"] = "evolink-key"
        config.app["evolink_base_url"] = ""
        config.app["evolink_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nevolink")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="evolink-key",
            base_url="https://direct.evolink.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nevolink")

    def test_volcengine_provider_uses_openai_compatible_client(self):
        """
        VolcEngine Ark  OpenAI-compatible Chat Completions. 
         fake OpenAI client  provider , 
         API key . 
        """
        config.app["llm_provider"] = "volcengine"
        config.app["volcengine_api_key"] = "volcengine-key"
        config.app["volcengine_base_url"] = ""
        config.app["volcengine_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nvolcengine")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="volcengine-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "doubao-seed-2-1-turbo-260628",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nvolcengine")

    def test_grok_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "grok"
        config.app["grok_api_key"] = ""
        config.app["grok_base_url"] = "https://api.x.ai/v1"
        config.app["grok_model_name"] = "grok-4.3"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_requires_api_key(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = ""
        config.app["groq_base_url"] = "https://api.groq.com/openai/v1"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_uses_default_base_url(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-test-key"
        config.app["groq_base_url"] = ""
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\ngroq")
                )
            ]
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(result, "hello\ngroq")

    def _use_ollama_provider(self, base_url=""):
        config.app["llm_provider"] = "ollama"
        config.app["ollama_api_key"] = ""
        config.app["ollama_base_url"] = base_url
        config.app["ollama_model_name"] = "llama3"

    def _assert_ollama_base_url(self, expected_base_url: str):
        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nollama")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="ollama",
            base_url=expected_base_url,
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nollama")

    def test_ollama_default_base_url_uses_localhost_outside_container(self):
        """
        , Ollama  localhost, . 
        """
        self._use_ollama_provider()

        with patch.object(config, "is_running_in_container", return_value=False):
            self._assert_ollama_base_url("http://localhost:11434/v1")

    def test_ollama_default_base_url_uses_host_gateway_inside_container(self):
        """
        , localhost ;  host.docker.internal, 
         Docker Desktop  Ollama. 
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=True),
        ):
            self._assert_ollama_base_url("http://host.docker.internal:11434/v1")

    def test_ollama_default_base_url_falls_back_to_container_gateway(self):
        """
         Linux Docker  host.docker.internal. 
        ,  hostname . 
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=False),
            patch.object(
                config, "get_container_default_gateway_ip", return_value="172.17.0.1"
            ),
        ):
            self._assert_ollama_base_url("http://172.17.0.1:11434/v1")

    def test_ollama_explicit_base_url_takes_precedence(self):
        """
         ollama_base_url , . 
        """
        self._use_ollama_provider(base_url="http://ollama:11434/v1")

        with patch.object(config, "is_running_in_container", return_value=True):
            self._assert_ollama_base_url("http://ollama:11434/v1")

    def test_mimo_provider_uses_openai_compatible_client(self):
        """
        MiMo  OpenAI Chat Completions .  fake OpenAI
        client  provider  MiMo  base_url, 
         API Key. 
        """
        config.app["llm_provider"] = "mimo"
        config.app["mimo_api_key"] = "mimo-key"
        config.app["mimo_base_url"] = ""
        config.app["mimo_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nmimo")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nmimo")

    def test_azure_provider_uses_azure_client_directly(self):
        """
        Azure OpenAI , endpoint  api-version  AzureOpenAI . 
         issue # This test covers issue #892: the azure branch must directly call the client created by AzureOpenAI,
         OpenAI-compatible ,  Azure . 
        """
        config.app["llm_provider"] = "azure"
        config.app["azure_api_key"] = "azure-key"
        config.app["azure_base_url"] = "https://example.openai.azure.com"
        config.app["azure_model_name"] = "gpt-4o-mini"
        config.app["azure_api_version"] = "2024-02-15-preview"

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nazure")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "AzureOpenAI", return_value=fake_client) as azure_client,
            patch.object(llm, "OpenAI") as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        azure_client.assert_called_once_with(
            api_key="azure-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://example.openai.azure.com",
        )
        openai_client.assert_not_called()
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nazure")

    def test_unsupported_provider_returns_clear_error(self):
        config.app["llm_provider"] = "g" + "4f"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("unsupported llm provider", result)


class TestRuntimeEnvironmentDetection(unittest.TestCase):
    def test_container_detection_ignores_plain_linux_cgroup_file(self):
        """
         Linux  /proc/1/cgroup, . 
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text("0::/init.scope\n", encoding="utf-8")

            self.assertFalse(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_detection_accepts_dockerenv_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerenv_path = Path(tmp_dir) / ".dockerenv"
            dockerenv_path.write_text("", encoding="utf-8")

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(dockerenv_path),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(Path(tmp_dir) / "missing-cgroup"),
                )
            )

    def test_container_detection_accepts_cgroup_container_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text(
                "0::/system.slice/docker-abcdef.scope\n",
                encoding="utf-8",
            )

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_gateway_ip_decodes_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)),
                "172.17.0.1",
            )

    def test_container_gateway_ip_ignores_missing_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t0011AC0A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)), ""
            )


class TestSocialMetadata(unittest.TestCase):
    """. """

    def test_build_prompt_auto_language_uses_source_language(self):
        """
        language  auto , , 
        ,  API . 
        """
        prompt = llm.build_social_metadata_prompt(
            video_subject="",
            video_script=". ",
            language="auto",
            platform="tiktok",
        )

        self.assertIn("TikTok", prompt)
        self.assertIn("Use the same language as the video subject and script", prompt)
        self.assertIn("", prompt)
        self.assertIn("array of exactly 5 strings", prompt)

    def test_build_prompt_accepts_explicit_language(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="Coffee tips",
            language="en-US",
            platform="youtube_shorts",
        )

        self.assertIn("YouTube Shorts", prompt)
        self.assertIn('Write "title" and "caption" in this language: en-US', prompt)
        self.assertIn("array of exactly 3 strings", prompt)

    def test_unknown_platform_falls_back_to_tiktok(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x",
            platform="unsupported-platform",
        )

        self.assertIn("TikTok", prompt)

    def test_normalize_hashtags_from_string_dedupes_and_clamps(self):
        tags = llm._normalize_hashtags("#fyp fyp, trending #Trending viral", count=2)

        self.assertEqual(tags, ["#fyp", "#trending"])

    def test_normalize_hashtags_from_list_keeps_unicode_letters(self):
        tags = llm._normalize_hashtags(
            ["#travel", "#việt nam", "@bad!chars"], count=5
        )

        self.assertEqual(tags, ["#travel", "#việtnam", "#badchars"])

    def test_parse_social_metadata_recovers_embedded_json(self):
        raw = 'Sure: {"title":"T","caption":"C","hashtags":["#x"]} thanks'
        result = llm._parse_social_metadata(raw, "tiktok")

        self.assertEqual(result["title"], "T")
        self.assertEqual(result["caption"], "C")
        self.assertEqual(result["hashtags"], ["#x"])

    def test_parse_social_metadata_requires_title_or_caption(self):
        with self.assertRaises(ValueError):
            llm._parse_social_metadata('{"hashtags":["#x"]}', "tiktok")

    def test_generate_social_metadata_uses_llm_response(self):
        payload = (
            '{"title":"","caption":", ! ",'
            '"hashtags":["# ","#","#shorts"]}'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_social_metadata(
                video_subject="",
                video_script=". ",
                language="zh-CN",
                platform="tiktok",
            )

        self.assertEqual(result["title"], "")
        self.assertEqual(result["caption"], ", ! ")
        self.assertEqual(result["hashtags"], ["# ", "#", "#shorts"])

    def test_generate_social_metadata_falls_back_to_generic_hashtags(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            result = llm.generate_social_metadata(
                video_subject="Coffee tips",
                video_script="Save these three coffee tips.",
                platform="instagram_reels",
            )

        self.assertEqual(result["title"], "Coffee tips")
        self.assertEqual(result["caption"], "Save these three coffee tips.")
        self.assertEqual(len(result["hashtags"]), 8)
        self.assertEqual(result["hashtags"][0], "#shorts")

    def test_request_model_defaults_to_auto_language_tiktok(self):
        body = VideoSocialMetadataRequest(video_subject="Test")

        self.assertEqual(body.language, "auto")
        self.assertEqual(body.platform, "tiktok")

    def test_request_model_rejects_oversized_social_metadata_fields(self):
        """
         API ,  LLM
        token . schema , . 
        """
        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x" * 501)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", video_script="x" * 8001)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", language="x" * 65)

    def test_build_prompt_clamps_direct_service_inputs(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x" * 600,
            video_script="y" * 9000,
            language="en",
        )

        self.assertIn("x" * llm.MAX_SOCIAL_SUBJECT_LENGTH, prompt)
        self.assertNotIn("x" * (llm.MAX_SOCIAL_SUBJECT_LENGTH + 1), prompt)
        self.assertIn("y" * llm.MAX_SOCIAL_SCRIPT_LENGTH, prompt)
        self.assertNotIn("y" * (llm.MAX_SOCIAL_SCRIPT_LENGTH + 1), prompt)

    def test_social_metadata_endpoint_response_shape(self):
        from fastapi.testclient import TestClient

        from app.asgi import app

        request_body = {
            "video_subject": "Tokyo coffee shops",
            "video_script": "Three quiet coffee shops for your next Tokyo morning.",
            "language": "en",
            "platform": "youtube_shorts",
        }
        llm_response = (
            '{"title":"3 Quiet Tokyo Coffee Shops",'
            '"caption":"Save these spots for your next Tokyo morning.",'
            '"hashtags":["#Tokyo","#Coffee","#Shorts"]}'
        )

        with patch.object(llm, "_generate_response", return_value=llm_response):
            response = TestClient(app).post(
                "/api/v1/social-metadata",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "3 Quiet Tokyo Coffee Shops",
                    "caption": "Save these spots for your next Tokyo morning.",
                    "hashtags": ["#Tokyo", "#Coffee", "#Shorts"],
                },
            },
        )


FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"
FOUNDRY_MODEL = "azure_ai/claude-sonnet-4-6"


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and FOUNDRY_KEY,
    "MPT_RUN_INTEGRATION_TESTS and ANTHROPIC_FOUNDRY_API_KEY not set",
)
class TestLiteLLMLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = FOUNDRY_MODEL
        os.environ["AZURE_AI_API_KEY"] = FOUNDRY_KEY
        os.environ["AZURE_AI_API_BASE"] = FOUNDRY_BASE

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_live_litellm_completion(self):
        result = llm._generate_response("What is 2+2? Reply with just the number.")

        self.assertNotIn("Error:", result)
        self.assertIn("4", result)


if __name__ == "__main__":
    unittest.main()
