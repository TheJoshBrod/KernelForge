import argparse
import json
import os
import time


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
    if not apikey:
        print(json.dumps({"success": False, "error": "Missing API key"}))
        return 1

    _set_env(provider, model, apikey)

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

    result.update({"success": True, "provider": provider, "model": model})
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
