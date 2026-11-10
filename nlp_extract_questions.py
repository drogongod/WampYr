"""Extract grounded Q&A pairs from the NLP/persuasion pilot corpus, same shape as the literature
pilot's questions file (literature_questions.jsonl), so lucy_diary_shakedown.py can consume the
output unchanged via --questions-file/--results-file/--checkpoint.

Chunking/output schema intentionally mirrors the (now-missing) literature extractor's observed
output: {question, ground_truth_answer, evidence_quote, category, source_file, section_header,
extractor_model, extracted_at}, one JSON object per line. Checkpoint format also mirrors it:
{"done": ["<source_file>::<chunk_idx>::<10-hex-hash>", ...], "timestamp": ...} so a kill/resume
never re-extracts a chunk already written to the questions file.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
HOST_EXE = Path(r"C:\llama.cpp\llama-server.exe")
HOST_MODEL_GGUF = Path(r"F:\LLM\Ornith-1.5-9B-uncensored-GGUF\Ornith-1.5-9B-uncensored.Q8_0.gguf")
HOST_URL = "http://127.0.0.1:8102"
HOST_ALIAS = "vampire-lucywest"
CHUNK_CHARS = 2800
QUESTIONS_PER_CHUNK = 6


def _host_ready():
    try:
        with urllib.request.urlopen(HOST_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_host():
    if _host_ready():
        with urllib.request.urlopen(HOST_URL + "/v1/models", timeout=5) as r:
            models = json.load(r).get("data", [])
        if not any(HOST_ALIAS == m.get("id") for m in models):
            raise RuntimeError("Port 8102 is occupied by a different model; no process was stopped.")
        return None
    if not HOST_EXE.is_file() or not HOST_MODEL_GGUF.is_file():
        raise FileNotFoundError(f"Host binary or model missing: {HOST_EXE}, {HOST_MODEL_GGUF}")
    log_dir = ENGINE_DIR / "data" / "server"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"server_{uuid.uuid4().hex}.log"
    args = [str(HOST_EXE), "-m", str(HOST_MODEL_GGUF), "--host", "127.0.0.1", "--port", "8102",
            "--alias", HOST_ALIAS, "-c", "8192", "-np", "1", "-ngl", "99", "--reasoning", "auto",
            "--no-mmap"]
    with log_path.open("xb") as f:
        proc = subprocess.Popen(args, cwd=ENGINE_DIR, stdout=f, stderr=f,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    print(f"Starting host. Log: {log_path}", flush=True)
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"Host exited; inspect {log_path}")
        if _host_ready():
            print(f"Host ready: {HOST_URL}", flush=True)
            return proc
        time.sleep(1)
    raise RuntimeError(f"Host still loading; inspect {log_path}. It was not stopped.")


def chunk_text(text, max_chars=CHUNK_CHARS):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


EXTRACT_SYSTEM = (
    "You extract grounded factual question-answer pairs from reference material for a study "
    "corpus. For the given passage, write up to {n} distinct question-answer pairs about its "
    "actual content (techniques, claims, definitions, examples, mechanisms it describes) -- not "
    "about the passage's existence or structure. Each pair needs a short, direct question; a "
    "concise 1-2 sentence answer; and an evidence_quote copied VERBATIM from the passage that "
    "supports the answer (exact substring, no paraphrasing, no ellipsis-joining of separate "
    "sentences). If the passage has fewer than {n} distinct extractable facts, return fewer -- "
    "never invent facts or quotes not present in the passage. Respond with ONLY a JSON array, no "
    "prose, no markdown fences: "
    '[{{"question": "...", "ground_truth_answer": "...", "evidence_quote": "..."}}, ...]'
)


def extract_json_array(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in host response")
    return json.loads(raw[start:end + 1])


def ask_host_for_pairs(passage, n=QUESTIONS_PER_CHUNK, max_tokens=3200):
    body = {
        "model": HOST_ALIAS,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM.format(n=n)},
            {"role": "user", "content": passage},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        HOST_URL + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    choice = resp.get("choices", [{}])[0]
    content = (choice.get("message", {}) or {}).get("content", "")
    if not content.strip() and choice.get("finish_reason") == "length":
        # Reasoning model burned the whole token budget on reasoning_content and never emitted
        # the actual answer -- same failure mode documented in server.py's history with the 27B
        # model. Retrying with the same budget reproduces the same failure (it's not noise, long
        # reasoning chains are near-deterministic at temperature=0.3); the fix is a bigger budget,
        # not a retry.
        raise TimeoutError(f"host exhausted {max_tokens} tokens on reasoning_content, empty content")
    return extract_json_array(content)


def load_checkpoint(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done": [], "timestamp": None}


def save_checkpoint(path, ckpt):
    ckpt["timestamp"] = time.time()
    path.write_text(json.dumps(ckpt), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-dir", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="source filenames (relative to --sources-dir) to extract")
    ap.add_argument("--questions-file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--category", default="nlp")
    args = ap.parse_args()

    sources_dir = Path(args.sources_dir)
    questions_path = Path(args.questions_file)
    checkpoint_path = Path(args.checkpoint)
    ckpt = load_checkpoint(checkpoint_path)
    done = set(ckpt["done"])

    ensure_host()

    total_written = 0
    for fname in args.files:
        fpath = sources_dir / fname
        text = fpath.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text)
        print(f"{fname}: {len(chunks)} chunks", flush=True)
        for idx, chunk in enumerate(chunks):
            chash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:10]
            key = f"{fname}::{idx}::{chash}"
            if key in done:
                continue
            pairs, last_exc = None, None
            for attempt, budget in enumerate((3200, 5000, 6500)):
                try:
                    pairs = ask_host_for_pairs(chunk, max_tokens=budget)
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(2)
            if pairs is None:
                print(f"  [{idx}/{len(chunks)}] ERROR after 3 attempts: {type(last_exc).__name__}: {last_exc}", flush=True)
                continue
            written_this_chunk = 0
            with questions_path.open("a", encoding="utf-8") as out:
                for pair in pairs:
                    q = (pair.get("question") or "").strip()
                    a = (pair.get("ground_truth_answer") or "").strip()
                    ev = (pair.get("evidence_quote") or "").strip()
                    if not q or not a or not ev:
                        continue
                    if ev not in chunk:
                        # evidence_quote must be a real verbatim substring -- drop anything the
                        # host hallucinated rather than silently trusting it.
                        continue
                    rec = {
                        "question": q,
                        "ground_truth_answer": a,
                        "evidence_quote": ev,
                        "category": args.category,
                        "source_file": fname,
                        "section_header": f"chunk_{idx}",
                        "extractor_model": HOST_ALIAS,
                        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written_this_chunk += 1
            total_written += written_this_chunk
            done.add(key)
            ckpt["done"] = list(done)
            save_checkpoint(checkpoint_path, ckpt)
            print(f"  [{idx + 1}/{len(chunks)}] +{written_this_chunk} questions (total {total_written})", flush=True)

    print(f"DONE. total questions written this run: {total_written}", flush=True)


if __name__ == "__main__":
    main()
