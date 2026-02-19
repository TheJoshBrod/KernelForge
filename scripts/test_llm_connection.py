import argparse
import json
import os
import sys
import time

# Ensure we can import from src
sys.path.append(os.getcwd())

try:
    from src.llm_tools import GenModel
except ImportError:
    # Fallback if src not found (e.g. running from wrong dir)
    print(json.dumps({"success": False, "error": "Could not import src.llm_tools"}))
    sys.exit(1)

def test_connection(provider, model, apikey):
    # Set env vars for the specific provider
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = apikey
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = apikey
    elif provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = apikey
        os.environ["GEMINI_API_KEY"] = apikey
    
    start_time = time.time()
    try:
        gen = GenModel("You are a helpful assistant. Reply with 'OK'.")
        response = gen.chat("Hello, are you there?", model)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        if not response:
            return {"success": False, "error": "Empty response from model"}
            
        return {
            "success": True, 
            "message": f"Response: {response[:20]}...", 
            "latency_ms": duration_ms,
            "provider": provider,
            "model": model
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "provider": provider}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--apikey", required=True)
    args = parser.parse_args()

    # Create dummy output directory if needed by some internal logic, though GenModel shouldn't need it
    # Just run test
    result = test_connection(args.provider, args.model, args.apikey)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
