#!/usr/bin/env python3
"""Interactive terminal chat with Lucy's live BrickCluster API.

Generic client, no identity-specific content -- works unchanged against either the Private or
Public server (her personality/system prompt lives server-side, driven by whichever engine
config is running). Talks to http://127.0.0.1:<port>/v1/chat/completions.

TRAINING, matching the real architecture (brickcluster_engine.py's own design, not a shortcut
added here): every turn is donor-assisted and logged as an UNTRUSTED pending experience
(experience_requires_acceptance=True) -- it does NOT silently write into her weights. After each
reply this script asks whether to accept it into her weights (the real /v1/feedback call,
the same mechanism amelia_teaches_lucy.py uses in bulk, just one human-reviewed turn at a time
here instead of auto-accept). Answer 'y' to teach her that turn, anything else to skip it.
"""
import json
import sys
import urllib.request
import urllib.error
import uuid

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8105
BASE = f"http://127.0.0.1:{PORT}"


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"[cannot reach Lucy's API at {BASE}: {exc}]")
        print("[start her server first -- see serve-lucy-brickcluster.bat]")
        sys.exit(1)
    print(f"[connected: {health['gateway_model']}  n_bricks={health['n_bricks']}  "
          f"checkpoint={health['checkpoint']}]")
    print("[type your message, Ctrl+C to quit; after each reply you'll be asked to accept it "
          "into her weights]\n")

    session_id = uuid.uuid4().hex
    while True:
        try:
            text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break
        if not text:
            continue
        try:
            response = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": text}],
                "stream": False, "vampire_session_id": session_id,
            })
        except urllib.error.HTTPError as exc:
            print(f"[error {exc.code}: {exc.read().decode('utf-8', 'replace')}]")
            continue
        except urllib.error.URLError as exc:
            print(f"[connection error: {exc}]")
            continue
        route = response.get("vampire_route", {})
        answer = response["choices"][0]["message"]["content"]
        print(f"lucy> {answer}")
        print(f"      [own-weights matched={route.get('lucy_brick_matched_existing')}  "
              f"n_bricks={route.get('n_bricks')}  source={route.get('source')}]")
        if not route.get("experience_requires_acceptance"):
            print("      [answered from her own weights -- no donor asked, nothing to accept]")
            continue
        try:
            accept = input("      teach her this turn? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break
        if accept == "y":
            try:
                result = post("/v1/feedback", {
                    "session_id": route.get("session_id", session_id),
                    "experience_id": route.get("experience_id"),
                    "action": "accept",
                })
                update = result.get("weight_update", {})
                print(f"      [learned: brick={update.get('brick_index')} "
                      f"new_brick={update.get('is_new_brick')} "
                      f"changed_levels={update.get('changed_levels')} "
                      f"n_bricks={update.get('n_bricks')}]")
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                print(f"      [accept failed: {exc}]")


if __name__ == "__main__":
    main()
