import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Ensure we can import from src if needed, though this script might be standalone
sys.path.append(os.getcwd())

def fetch_openai_models(api_key: str) -> List[Dict[str, Any]]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        
        if models and hasattr(models, 'data'):
            # Filter for pure chat/reasoning models.
            # Exclude audio, realtime, voice, tts, transcribe, search-api, image
            
            chat_models = []
            for m in models.data:
                mid = m.id.lower()
                # Must start with gpt-, o1-, or o3-
                if not (mid.startswith("gpt-") or mid.startswith("o1-") or mid.startswith("o3-")):
                    continue
                
                # Exclude unwanted types
                if "audio" in mid or "realtime" in mid or "voice" in mid:
                    continue
                if "tts" in mid or "transcribe" in mid or "search-api" in mid:
                    continue
                if "image" in mid or "dall-e" in mid:
                    continue
                
                chat_models.append({
                    "id": m.id,
                    "created": m.created,
                    "provider": "openai",
                    "label": m.id
                })
            return chat_models
        return []
    except Exception as e:
        # print(f"OpenAI Error: {e}", file=sys.stderr)
        return []

def fetch_anthropic_models(api_key: str) -> List[Dict[str, Any]]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Anthropic list models API is now available but might be limited in older SDK versions
        # Documentation says GET /v1/models
        # For now, if the SDK supports it, use it. Otherwise, return a hardcoded recent list as fallback?
        # Let's try to list.
        
        # Newer Anthropic SDKs might support list.
        # If not, we might need a fallback list or just return empty if strictly dynamic.
        # But let's check if client.models.list() exists
        
        models_page = client.models.list()
        
        ant_models = []
        for m in models_page.data:
             # Anthropic models have 'id', 'display_name', 'created_at' usually?
             # Checking standard response structure: { id: "claude-3-...", ... }
             # Note: created_at might be missing in some versions/endpoints.
             created = getattr(m, 'created_at', 0)
             # Start date for some known models if 0?
             
             ant_models.append({
                 "id": m.id,
                 "created": 0, # Models endpoint might not give creation date easily, defaulting
                 "provider": "anthropic",
                 "label": m.display_name if hasattr(m, 'display_name') else m.id
             })
             
        return ant_models

    except Exception as e:
        # print(f"Anthropic Error: {e}", file=sys.stderr)
        # Fallback list if API fails or method doesn't exist (common with older SDKs or permissions)
        # But for dynamic requirement, we return empty if failed.
        return []

def fetch_gemini_models(api_key: str) -> List[Dict[str, Any]]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # models.list returns an iterator of Model objects
        # We need to iterate it.
        # Check standard list response.
        
        # Create a list from the iterator
        all_models = list(client.models.list())
        
        gem_models = []
        for m in all_models:
             # m.name is full resource name e.g. "models/gemini-1.5-flash"
             # m.display_name is "Gemini 1.5 Flash"
             # m.supported_generation_methods usually includes "generateContent"
             
             # Some versions/models might return different structures.
             # Let's be permissive if name contains 'gemini'
             
             model_name = getattr(m, 'name', '')
             if 'gemini' in model_name.lower():
                 # Check generation methods if available
                 methods = getattr(m, 'supported_generation_methods', [])
                 if methods and 'generateContent' not in methods:
                     continue
                 
                 # Exclude unwanted Gemini models
                 mid = model_name.lower()
                 if "embedding" in mid: continue
                 if "image" in mid: continue
                 if "vision" in mid and "gemini-pro-vision" not in mid: continue
                 # Actually gemini-pro-vision is chat. 
                 # Exclude specific non-chat experiential
                 if "robotics" in mid: continue
                 if "computer-use" in mid: continue
                 if "tts" in mid or "audio" in mid: continue
                 
                 clean_id = model_name.replace("models/", "")
                 gem_models.append({
                     "id": clean_id,
                     "created": 0, 
                     "provider": "gemini",
                     "label": getattr(m, 'display_name', clean_id)
                 })
                 
        return gem_models

    except Exception as e:
        # print(f"Gemini Error: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", default="")
    parser.add_argument("--anthropic_key", default="")
    parser.add_argument("--gemini_key", default="")
    args = parser.parse_args()

    all_models = []

    if args.openai_key:
        all_models.extend(fetch_openai_models(args.openai_key))
    
    if args.anthropic_key:
        all_models.extend(fetch_anthropic_models(args.anthropic_key))

    if args.gemini_key:
        all_models.extend(fetch_gemini_models(args.gemini_key))

    # Sort by created date descending (newest first)
    # For models with 0 created date (Anthropic/Gemini), they will be at bottom.
    # Maybe we want to interleave or sort by name for them?
    # User requested sort by release date. OpenAI has it. 
    # Let's simple sort desc.
    
    all_models.sort(key=lambda x: x.get("created", 0), reverse=True)

    print(json.dumps(all_models))

if __name__ == "__main__":
    main()
