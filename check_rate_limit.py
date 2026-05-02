"""Check OpenRouter rate limits for the models used in TradingAgents."""

import json
import os
import urllib.request

# Load .env manually (no dotenv dep needed)
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODELS = [
    "tencent/hy3-preview:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()), dict(resp.headers)


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200], dict(e.headers)


def check_account_limits():
    data, _ = _get("https://openrouter.ai/api/v1/key")
    d = data["data"]
    print("=== Account ===")
    print(f"  Credits remaining : ${d['limit_remaining']:.4f} / ${d['limit']}")
    print(f"  Usage (today)     : ${d['usage_daily']:.6f}")
    print()


def check_model_rate_limits():
    print("=== Per-model rate limits (via response headers) ===")
    for model in MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        }
        status, body, headers = _post(
            "https://openrouter.ai/api/v1/chat/completions", payload
        )

        rl_limit     = headers.get("x-ratelimit-limit-requests", "n/a")
        rl_remaining = headers.get("x-ratelimit-remaining-requests", "n/a")
        rl_reset     = headers.get("x-ratelimit-reset-requests", "n/a")

        ok = "OK" if status == 200 else f"HTTP {status}"
        print(f"  {model}")
        print(f"    status    : {ok}")
        print(f"    limit     : {rl_limit} req")
        print(f"    remaining : {rl_remaining} req")
        print(f"    resets in : {rl_reset}")
        if status != 200:
            print(f"    body      : {body}")
        print()


if __name__ == "__main__":
    check_account_limits()
    check_model_rate_limits()
