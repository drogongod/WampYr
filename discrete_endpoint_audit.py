"""DISCRETE ENDPOINT AUDIT (Astra's Priority 1, LUCYWEST_INDEPENDENT_TECHNICAL_HANDOFF_20261031.md
section 5) -- never run before in this investigation. Every one of the 7 negative mechanism results
tuned the UPDATE RULE (threshold, momentum, hysteresis, caps) without ever directly checking whether
the flips the rule actually chose were beneficial at the real, discrete, finite-difference loss.
This measures that directly: snapshot before/after one real learn_batch call, then for each weight
that ACTUALLY flipped, isolate it (restore everything else to pre-update, apply just that one flip)
and measure the real teacher-forced loss change -- no smoothing, no gradient proxy, the real forward
pass at the real discrete endpoint. Also measures the joint (all real flips together) vs the sum of
sampled individual effects, to get a real interaction term (avalanche test).

Also fixes a real methodology gap Astra caught (section 3.10): every CE number in this whole
investigation was computed over the FULL line (including the "User: ...?\n" question prefix), never
isolated to just the "Lucy: <answer>" portion -- reports both here, explicitly labeled.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, r"F:\LucyWEST\PrivateLucyWEST\engine")
import numpy as np
from brick_cluster_charlm import FixedStepBrick
from lutgtlm_ingest import _line_batches, enc

FACT = "User: What is the capital of France?\nLucy: The capital of France is Paris."
PROMPT_PREFIX = "User: What is the capital of France?\nLucy: "
ANSWER_START = len(enc(PROMPT_PREFIX))
RANDOM_BASELINE = np.log(256)
MATRICES = ["Win", "Wh", "Wgx", "Wgh", "Wo"]


def loss_per_token(brick, line):
    ids = enc(line)
    logits = brick.forward_logits(ids[:-1])
    targets = np.array(ids[1:])
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return -logp[np.arange(len(targets)), targets]


def full_line_nll(brick, line):
    return float(loss_per_token(brick, line).mean())


def answer_only_nll(brick, line):
    nll = loss_per_token(brick, line)
    start = max(0, ANSWER_START - 1)
    return float(nll[start:].mean())


def snapshot(brick):
    return {m: getattr(brick, m).copy() for m in MATRICES}


def restore(brick, snap):
    for m in MATRICES:
        setattr(brick, m, snap[m].copy())


def run_audit(brick, rep_label, pre, post):
    restore(brick, pre)
    base_full = full_line_nll(brick, FACT)
    base_ans = answer_only_nll(brick, FACT)
    restore(brick, post)
    after_full = full_line_nll(brick, FACT)
    after_ans = answer_only_nll(brick, FACT)
    print(f"\n===== {rep_label} =====")
    print(f"BEFORE this call's update: full-line NLL={base_full:.3f}  answer-only NLL={base_ans:.3f}  "
          f"(random baseline={RANDOM_BASELINE:.3f})")
    print(f"AFTER  this call's update: full-line NLL={after_full:.3f}  answer-only NLL={after_ans:.3f}")

    flips = []
    for m in MATRICES:
        diff = pre[m].astype(np.int16) != post[m].astype(np.int16)
        for ij in np.argwhere(diff):
            flips.append((m, tuple(ij)))
    print(f"Total real flips this call: {len(flips)}")
    if not flips:
        restore(brick, post)
        return

    rng = np.random.default_rng(0)
    sample = flips if len(flips) <= 80 else [flips[i] for i in rng.choice(len(flips), 80, replace=False)]

    deltas_full, deltas_ans = [], []
    for m, ij in sample:
        restore(brick, pre)
        arr = getattr(brick, m).copy()
        arr[ij] = post[m][ij]
        setattr(brick, m, arr)
        deltas_full.append(full_line_nll(brick, FACT) - base_full)
        deltas_ans.append(answer_only_nll(brick, FACT) - base_ans)
    restore(brick, pre)

    def tally(deltas):
        h = sum(1 for d in deltas if d < -1e-9)
        bad = sum(1 for d in deltas if d > 1e-9)
        neu = len(deltas) - h - bad
        return h, bad, neu

    n = len(sample)
    hf, bf, nf = tally(deltas_full)
    ha, ba, na = tally(deltas_ans)
    print(f"\nAUDIT A -- individual real flips, isolated, n={n} of {len(flips)} total flips:")
    print(f"  (full-line)   helpful={hf}/{n}  harmful={bf}/{n}  neutral={nf}/{n}  mean_delta={np.mean(deltas_full):+.4f}")
    print(f"  (answer-only) helpful={ha}/{n}  harmful={ba}/{n}  neutral={na}/{n}  mean_delta={np.mean(deltas_ans):+.4f}")

    restore(brick, pre)
    l0_full, l0_ans = base_full, base_ans
    restore(brick, post)
    l_group_full = full_line_nll(brick, FACT)
    l_group_ans = answer_only_nll(brick, FACT)
    sum_full = sum(deltas_full)
    sum_ans = sum(deltas_ans)
    interaction_full = (l_group_full - l0_full) - sum_full
    interaction_ans = (l_group_ans - l0_ans) - sum_ans
    print(f"\nAUDIT B -- joint (ALL {len(flips)} real flips together) vs sum of the {n} SAMPLED individual effects:")
    print(f"  (full-line)   group_delta={l_group_full-l0_full:+.4f}  sum_sampled={sum_full:+.4f}  interaction={interaction_full:+.4f}")
    print(f"  (answer-only) group_delta={l_group_ans-l0_ans:+.4f}  sum_sampled={sum_ans:+.4f}  interaction={interaction_ans:+.4f}")
    restore(brick, post)


brick = FixedStepBrick(V=256, D=40, seed=0)
print(f"random baseline (uniform over 256 bytes): {RANDOM_BASELINE:.3f} nats")
print(f"ANSWER_START byte index (where 'Lucy: ' ends): {ANSWER_START}")

for rep in range(1, 7):
    pre = snapshot(brick)
    for b in _line_batches(FACT):
        brick.learn_batch(b, lr_scale=1.0, corrob_eligible=False)
    post = snapshot(brick)
    run_audit(brick, f"REP {rep}", pre, post)
