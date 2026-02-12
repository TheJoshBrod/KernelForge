import argparse
import json
import os
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.auth.credentials import apply_auth_env, resolve_auth
from src.config import load_config_data


def _set_env(provider: str, model: str, apikey: str):
    provider = provider.strip().lower()
    os.environ["LLM_PROVIDER"] = provider
    if provider == "openai":
        if apikey:
            os.environ["OPENAI_API_KEY"] = apikey
        if model:
            os.environ["OPENAI_MODEL"] = model
    elif provider == "anthropic":
        if apikey:
            os.environ["ANTHROPIC_API_KEY"] = apikey
        if model:
            os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "gemini":
        if apikey:
            os.environ["GEMINI_API_KEY"] = apikey
            os.environ["GOOGLE_API_KEY"] = apikey
        if model:
            os.environ["GEMINI_MODEL"] = model


def _resolve_effective_auth(provider: str, model: str, apikey: str) -> dict:
    cfg, _ = load_config_data()
    cfg = dict(cfg or {})
    llm = cfg.get("llm_info") if isinstance(cfg.get("llm_info"), dict) else {}
    llm["provider"] = provider
    llm["model"] = model
    if apikey:
        llm["apikey"] = apikey
    cfg["llm_info"] = llm

    auth_cfg = cfg.get("auth") if isinstance(cfg.get("auth"), dict) else {}
    if provider:
        auth_cfg["provider"] = provider
    if model:
        auth_cfg["model"] = model
    cfg["auth"] = auth_cfg

    status = resolve_auth(
        config=cfg,
        env=dict(os.environ),
        runtime_context={"in_container": bool(os.environ.get("CGINS_PROJECT_DIR"))},
    )
    apply_auth_env(status, os.environ)
    return status.to_dict()


def _test_openai(model: str) -> dict:
    from openai import OpenAI

    client = OpenAI()
    use_responses = model.startswith("gpt-5")
    started = time.time()

    if use_responses:
        resp = client.responses.create(
            model=model,
            input="Respond with the single word: ok",
            max_output_tokens=16,
        )
        text = resp.output_text.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Respond with the single word: ok"}],
            max_tokens=16,
        )
        text = resp.choices[0].message.content.strip()

    return {
        "response": text,
        "latency_ms": int((time.time() - started) * 1000),
    }


def _test_anthropic(model: str) -> dict:
    try:
        from anthropic import Anthropic
    except Exception as e:
        raise RuntimeError(f"anthropic package not installed: {e}")

    client = Anthropic()
    started = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": "Respond with the single word: ok"}],
    )
    text = resp.content[0].text.strip() if resp.content else ""
    return {
        "response": text,
        "latency_ms": int((time.time() - started) * 1000),
    }


def _test_gemini(model: str) -> dict:
    try:
        from google import genai
    except Exception as e:
        raise RuntimeError(f"google-genai package not installed: {e}")

    client = genai.Client()
    started = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": "Respond with the single word: ok"}]}],
    )
    text = resp.text.strip() if resp and resp.text else ""
    return {
        "response": text,
        "latency_ms": int((time.time() - started) * 1000),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LLM API connectivity.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--apikey", default="")
    args = parser.parse_args()

    provider = args.provider.strip().lower()
    model = args.model.strip()
    apikey = args.apikey.strip()

    if not provider:
        print(json.dumps({"success": False, "error": "Missing provider"}))
        return 1
    if not model:
        print(json.dumps({"success": False, "error": "Missing model"}))
        return 1
    if apikey:
        _set_env(provider, model, apikey)
    status = _resolve_effective_auth(provider, model, apikey)

    if status.get("mode_effective") == "account_session" and provider == "openai":
        print(
            json.dumps(
                {
                    "success": True,
                    "provider": provider,
                    "model": model,
                    "mode_effective": "account_session",
                    "response": "account session detected",
                    "latency_ms": 0,
                }
            )
        )
        return 0

    try:
        if provider == "openai":
            result = _test_openai(model)
        elif provider == "anthropic":
            result = _test_anthropic(model)
        elif provider == "gemini":
            result = _test_gemini(model)
        else:
            print(json.dumps({"success": False, "error": f"Unsupported provider: {provider}"}))
            return 1
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

    result.update(
        {
            "success": True,
            "provider": provider,
            "model": model,
            "mode_effective": status.get("mode_effective", ""),
        }
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
