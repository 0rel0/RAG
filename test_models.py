"""
Reads models.json, tests each chat-capable model with a simple "hi" request,
and reports which ones work with the current MISTRAL_API_KEY.
"""
import json
import os
from pathlib import Path
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = Path(os.environ["MISTRAL_API_KEY_FILE"]).read_text().strip()
URL = "https://api.mistral.ai/v1/chat/completions"


def main():
    with open("models.json") as f:
        data = json.load(f)

    # only test models that support chat completion, skip duplicate aliases
    seen_billing_names = set()
    models_to_test = []
    for m in data["data"]:
        if not m["capabilities"]["completion_chat"]:
            continue
        billing_name = m["billing_model_name"]
        if billing_name in seen_billing_names:
            continue
        seen_billing_names.add(billing_name)
        models_to_test.append(m["id"])

    print(f"Testing {len(models_to_test)} unique chat models...\n")

    working = []
    failing = []

    for model_id in models_to_test:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(URL, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                working.append(model_id)
                print(f"✅ {model_id}")
            else:
                failing.append((model_id, resp.status_code, resp.text[:150]))
                print(f"❌ {model_id} -> {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            failing.append((model_id, "exception", str(e)))
            print(f"⚠️ {model_id} -> exception: {e}")

        time.sleep(1.2)  # avoid tripping rate limits between requests

    print("\n--- SUMMARY ---")
    print(f"Working models ({len(working)}):")
    for m in working:
        print(f"  - {m}")
    print(f"\nFailing models ({len(failing)}):")
    for m, code, msg in failing:
        print(f"  - {m}: {code}")


if __name__ == "__main__":
    main()