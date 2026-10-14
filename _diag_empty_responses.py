#!/usr/bin/env python3
"""Direct diagnostic: call the teacher endpoint repeatedly with the REAL category-forcing prompt
structure, capturing the FULL raw response (finish_reason, reasoning_content length, content
length) for both successful and empty/truncated cases -- to see the actual mechanism, not guess
at it."""
import sys, json
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import amelia_teaches_lucy as A
import brickcluster_engine as E

CATEGORIES = A.CATEGORIES
asked = []

for i in range(20):
    category = CATEGORIES[i % len(CATEGORIES)]
    body = A.build_teacher_body(asked, category)  # the real, current body -- no duplicated prompt
    response = E.post_json(E.REFERENCE_URL.rstrip("/") + "/v1/chat/completions", body, timeout=120)
    msg = response["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish_reason = response["choices"][0].get("finish_reason")
    usage = response.get("usage", {})
    print(f"[{i}] category={category!r}")
    print(f"    finish_reason={finish_reason} completion_tokens={usage.get('completion_tokens')} "
          f"reasoning_len={len(reasoning)} content_len={len(content)}")
    if content:
        print(f"    content: {content[:150]!r}")
        asked.append(content.strip().strip('"').strip())
    else:
        print(f"    EMPTY CONTENT. reasoning tail: {reasoning[-300:]!r}")
    print()
