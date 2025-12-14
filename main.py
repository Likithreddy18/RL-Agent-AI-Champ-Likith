"""
Grocery demand forecasting RL task (single store).
Key properties: rolling-origin scoring, baseline-relative pass, enforced leakage trap,
subprocess execution with hard timeout, determinism, and prediction sanity bounds.
"""

import asyncio, json, math, multiprocessing as mp, random, sys, time
from typing import Any, Callable, Dict, List, Tuple

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, ToolUnionParam
except Exception:  # allows local grading without Anthropic installed
    AsyncAnthropic = None
    MessageParam = Any
    ToolUnionParam = Any

# Knobs (tune IMPROVEMENT_REQ to hit 10-40% pass rate)
HORIZON, HISTORY, TOTAL_DAYS, N_CUTOFFS = 14, 365, 730, 8
IMPROVEMENT_REQ, LEAK_SENS_MAX = 0.46, 0.02  
RUNTIME_BUDGET_S, SUBPROC_TIMEOUT_S = 30.0, 10.0
PRED_MAX_MULT = 3.0
MODEL_NAME = "claude-sonnet-4-20250514"

def mean(xs: List[float]) -> float: return sum(xs) / max(1, len(xs))
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def wape(y: List[float], p: List[float]) -> float:
    d = sum(max(0.0, float(v)) for v in y)
    if d <= 1e-9: return 0.0
    return 100.0 * sum(abs(float(a) - float(b)) for a, b in zip(y, p)) / d

def generate_series(seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    weekly = [1.00, 0.97, 0.94, 1.00, 1.08, 1.28, 1.18]
    base, promo_lift, hol_lift = 180.0, 1.18, 1.35
    hol_days, d = set(), 25
    while d < TOTAL_DAYS: hol_days.add(d); d += rng.randint(45, 70)

    sales, promo, hol = [], [], []
    for t in range(TOTAL_DAYS):
        dow = t % 7
        annual = 1.0 + 0.08 * math.sin(2.0 * math.pi * t / 365.0)
        trend = 1.0 + 0.00035 * t
        if t == 0: p = 1 if rng.random() < 0.10 else 0
        else:
            prev = promo[-1]
            p = 1 if (prev == 1 and rng.random() < 0.65) else (1 if rng.random() < 0.12 else 0)
        h = 1 if t in hol_days else 0
        noise = 1.0 + rng.gauss(0.0, 0.07)
        level = base * weekly[dow] * annual * trend * noise
        if p: level *= promo_lift
        if h: level *= hol_lift
        sales.append(int(max(0.0, level))); promo.append(p); hol.append(h)

    leak = []
    for t in range(TOTAL_DAYS):
        leak.append((sales[t + 1] + rng.randint(-3, 3)) if t < TOTAL_DAYS - 1 else int(mean([float(v) for v in sales[-7:]])))
    return {"sales": sales, "promo": promo, "holiday": hol, "leak_proxy": leak}

def baseline_forecast(hs: List[int], hp: List[int], hh: List[int], fp: List[int], fh: List[int]) -> List[float]:
    n, look = len(hs), min(len(hs), 56)
    rs, rp, rh = hs[-look:], hp[-look:], hh[-look:]
    dow = []
    for k in range(7):
        vals = [float(rs[i]) for i in range(k, look, 7)]
        dow.append(mean(vals) if vals else mean([float(v) for v in rs]))

    base0, base1, hol0, hol1 = [], [], [], []
    for y, p, h in zip(rs, rp, rh):
        y = float(y)
        (base1 if p else base0).append(y)
        (hol1 if h else hol0).append(y)

    pm = (mean(base1) / mean(base0)) if base1 and base0 else 1.0
    hm = (mean(hol1) / mean(hol0)) if hol1 and hol0 else 1.0
    pm, hm = clamp(pm, 1.0, 1.35), clamp(hm, 1.0, 1.60)

    out = []
    for k in range(HORIZON):
        y = dow[(n + k) % 7]
        if fp[k]: y *= pm
        if fh[k]: y *= hm
        out.append(max(0.0, float(y)))
    return out

def _worker(q: "mp.Queue", code: str, tasks: List[Dict[str, Any]]) -> None:
    try:
        safe = {"range": range, "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                "float": float, "int": int, "round": round, "enumerate": enumerate, "zip": zip, "list": list}
        ns: Dict[str, Any] = {"__builtins__": safe, "math": math}
        exec(code, ns, ns)
        fn = ns.get("forecast")
        if not callable(fn): raise ValueError("Missing forecast(...)")

        def clean(preds: Any) -> List[float]:
            if not isinstance(preds, list) or len(preds) != HORIZON: raise ValueError(f"Return list[{HORIZON}]")
            out = []
            for v in preds:
                fv = float(v)
                if not math.isfinite(fv) or fv < 0: raise ValueError("Predictions must be finite and non-negative")
                out.append(fv)
            return out

        t0 = tasks[0]
        a = clean(fn(t0["hs"], t0["hp"], t0["hh"], t0["hl"], t0["fp"], t0["fh"]))
        b = clean(fn(t0["hs"], t0["hp"], t0["hh"], t0["hl"], t0["fp"], t0["fh"]))
        if any(abs(x - y) > 1e-9 for x, y in zip(a, b)): raise ValueError("Non-deterministic output")

        perm = t0["hl"][:]; random.Random(1337).shuffle(perm)
        zero = [0] * len(t0["hl"])
        ap = clean(fn(t0["hs"], t0["hp"], t0["hh"], perm, t0["fp"], t0["fh"]))
        az = clean(fn(t0["hs"], t0["hp"], t0["hh"], zero, t0["fp"], t0["fh"]))
        eps = 1e-6
        leak = max(mean([abs(x - y) / max(eps, abs(x)) for x, y in zip(a, ap)]),
                   mean([abs(x - y) / max(eps, abs(x)) for x, y in zip(a, az)]))

        preds_all = [clean(fn(t["hs"], t["hp"], t["hh"], t["hl"], t["fp"], t["fh"])) for t in tasks]
        q.put({"ok": True, "preds": preds_all, "leak": float(leak), "error": None})
    except Exception as e:
        q.put({"ok": False, "preds": None, "leak": None, "error": str(e)})

def run_candidate(code: str, tasks: List[Dict[str, Any]]) -> Tuple[List[List[float]], float]:
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, code, tasks), daemon=True)
    p.start(); p.join(timeout=SUBPROC_TIMEOUT_S)
    if p.is_alive():
        p.terminate(); p.join(timeout=0.2)
        raise TimeoutError("Candidate timed out")
    if q.empty(): raise RuntimeError("No result from candidate")
    res = q.get()
    if not res.get("ok"): raise RuntimeError(res.get("error") or "Candidate failed")
    return res["preds"], float(res["leak"])

def score_code_on_seed(code: str, seed: int) -> Tuple[float, float, float]:
    t0 = time.time()
    d = generate_series(seed)
    sales, promo, hol, leak = d["sales"], d["promo"], d["holiday"], d["leak_proxy"]

    last = TOTAL_DAYS - HORIZON - 1
    first = max(HISTORY, TOTAL_DAYS - (N_CUTOFFS * 35) - HORIZON - 1)
    cutoffs, t = [], first
    while len(cutoffs) < N_CUTOFFS and t <= last: cutoffs.append(t); t += 35

    tasks, y_true, base_preds = [], [], []
    for c in cutoffs:
        hs, hp, hh, hl = sales[c - HISTORY:c], promo[c - HISTORY:c], hol[c - HISTORY:c], leak[c - HISTORY:c]
        fp, fh, yt = promo[c:c + HORIZON], hol[c:c + HORIZON], sales[c:c + HORIZON]
        tasks.append({"hs": hs, "hp": hp, "hh": hh, "hl": hl, "fp": fp, "fh": fh})
        y_true.append([float(x) for x in yt])
        base_preds.append(baseline_forecast(hs, hp, hh, fp, fh))

    if time.time() - t0 > RUNTIME_BUDGET_S: raise TimeoutError("Budget exceeded pre-candidate")

    cand_preds, leak_sens = run_candidate(code, tasks)

    base_scores, cand_scores = [], []
    for i, (yt, bp, cp) in enumerate(zip(y_true, base_preds, cand_preds)):
        mh = max(1.0, float(max(tasks[i]["hs"])))
        if any(v > PRED_MAX_MULT * mh for v in cp): raise ValueError("Prediction exceeds sanity bounds")
        base_scores.append(wape(yt, bp)); cand_scores.append(wape(yt, cp))

    if time.time() - t0 > RUNTIME_BUDGET_S: raise TimeoutError("Budget exceeded post-candidate")
    return mean(base_scores), mean(cand_scores), float(leak_sens)

def score_dev_tool(payload: dict) -> dict:
    out = {"baseline_wape": None, "candidate_wape": None, "improvement": None, "leak_sensitivity": None, "error": None}
    try:
        b, c, ls = score_code_on_seed(payload.get("code", ""), seed=7)
        out["baseline_wape"], out["candidate_wape"], out["leak_sensitivity"] = b, c, ls
        out["improvement"] = (b - c) / max(1e-9, b)
    except Exception as e:
        out["error"] = str(e)
    return out

def submit_tool(payload: dict) -> dict: return {"submitted": True, "code": payload.get("code", "")}

def build_prompt() -> str:
    return f"""# Grocery Demand Forecasting RL Task

Write Python: forecast(history_sales, history_promos, history_holidays, history_leak_proxy, future_promos, future_holidays) -> list[{HORIZON}]

## Data Patterns (model these to beat baseline)
- Weekly seasonality (weekends higher), Annual seasonality (sine wave)
- Promo effect (~18% lift), Holiday effect (~35% lift), Mild upward trend

## Inputs
history_sales/promos/holidays: {HISTORY} days of past data | history_leak_proxy: TRAP, don't use!
future_promos/holidays: next {HORIZON} days (known in advance)

## Constraints (grader-enforced)
- No leakage: leak_proxy is a trap. If predictions change >{int(LEAK_SENS_MAX*100)}% when permuted/zeroed, you fail.
- Determinism: same inputs must produce same outputs.
- Output: list[{HORIZON}]; finite; non-negative; each pred <= {PRED_MAX_MULT}*max(history_sales).

## Scoring
- Rolling-origin eval across {N_CUTOFFS} cutoffs. Metric: mean WAPE.
- Must beat baseline by >={int(IMPROVEMENT_REQ*100)}% on WAPE.

## Tools
- score_dev: test code, returns WAPE metrics. - submit: submit final code.
"""

TOOLS: List[ToolUnionParam] = [
    {"name": "score_dev", "description": "Score candidate code. Input: {'code': <python string>}.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "submit", "description": "Submit final code. Input: {'code': <python string>}.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
]
HANDLERS: Dict[str, Callable[[dict], dict]] = {"score_dev": score_dev_tool, "submit": submit_tool}

async def run_agent(prompt: str, max_steps: int = 12) -> str | None:
    if AsyncAnthropic is None: raise RuntimeError("Anthropic SDK not installed")
    client = AsyncAnthropic()
    messages: List[MessageParam] = [{"role": "user", "content": prompt}]
    for _ in range(max_steps):
        resp = await client.messages.create(model=MODEL_NAME, max_tokens=4096, tools=TOOLS, messages=messages)
        tool_results, submitted = [], None
        for c in resp.content:
            if c.type != "tool_use": continue
            out = HANDLERS.get(c.name, lambda _: {"error": "unknown tool"})(c.input)
            if c.name == "submit": submitted = out.get("code", "")
            tool_results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out)})
        if not tool_results: break
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})
        if submitted: return submitted
    return None

def grade(code: str, seed: int) -> Tuple[bool, str]:
    if not isinstance(code, str) or len(code.strip()) < 20: return False, "Missing or too-short code"
    try:
        b, c, ls = score_code_on_seed(code, seed=seed)
    except Exception as e:
        return False, f"Runtime error: {e}"
    imp = (b - c) / max(1e-9, b)
    if ls > LEAK_SENS_MAX: return False, f"Leakage detected: {ls:.3f} > {LEAK_SENS_MAX:.3f}"
    if imp < IMPROVEMENT_REQ:
        return False, f"Insufficient improvement: {imp:.1%} (need {IMPROVEMENT_REQ:.1%}). Baseline={b:.2f}, Candidate={c:.2f}"
    return True, f"PASS: Baseline={b:.2f}, Candidate={c:.2f}, Improvement={imp:.1%}, LeakSens={ls:.3f}"

async def run_one(i: int, n: int, seed: int) -> bool:
    code = await run_agent(build_prompt())
    ok, msg = (False, "No submission") if code is None else grade(code, seed)
    print(f"Run {i}/{n}: {'OK' if ok else 'FAIL'} | {msg}")
    return ok

async def run_single() -> None:
    """Single run mode - useful for testing with limited API quota."""
    print("Running single API test...\n")
    code = await run_agent(build_prompt())
    if code is None:
        print("FAIL: No code submitted by agent"); return
    print(f"Agent submitted code ({len(code)} chars)\n" + "-"*40)
    ok, msg = grade(code, seed=11)
    print(f"Result: {'PASS' if ok else 'FAIL'}\nDetails: {msg}")

async def run_full(few: bool = False) -> None:
    """Full evaluation mode - runs seeds sequentially to avoid rate limits."""
    all_seeds = [11, 23, 37, 41, 53, 67, 71, 89, 97, 101]
    seeds = all_seeds[:3] if few else all_seeds  # --few runs only 3 seeds
    n = len(seeds)
    print(f"Running evaluation ({n} seeds, sequential to avoid rate limits)...\n")
    
    results = []
    for i, seed in enumerate(seeds):
        ok = await run_one(i + 1, n, seed)
        results.append(ok)
        if i < n - 1:  # Don't wait after last run
            print("   (waiting 30s for rate limit...)")
            await asyncio.sleep(30)  # Wait between runs to avoid rate limit
    
    passes = sum(1 for r in results if r)
    pct = passes/n*100
    print(f"\n{'='*50}\nPass rate: {passes}/{n} ({pct:.0f}%)\n{'='*50}")
    print("✓ In target range (10-40%)" if 10 <= pct <= 40 else f"⚠ Outside range. Adjust IMPROVEMENT_REQ ({IMPROVEMENT_REQ})")

async def main() -> None:
    """Main entry point. Flags: --single (1 run), --few (3 runs), or no flag (10 runs)."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            await run_single()
        elif sys.argv[1] == "--few":
            await run_full(few=True)
        else:
            print("Usage: python main.py [--single | --few]")
    else:
        await run_full()

if __name__ == "__main__":
    asyncio.run(main())
