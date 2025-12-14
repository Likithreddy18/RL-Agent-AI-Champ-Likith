# Grocery Demand Forecasting - RL Training Task

This is a RL task for LLM training that tests real demand forecasting skills.

## Task Summary

| Property | Implementation |
|----------|---------------|
| **Evaluation** | Rolling-origin across 8 cutoffs on 2-year synthetic series |
| **Grading** | Baseline-relative: must beat by ≥12% on mean WAPE |
| **Leakage trap** | Dual counterfactual (permute + zero `leak_proxy`) |
| **Isolation** | Subprocess with hard timeout |
| **Bounds** | Predictions must be ≤3× max(history) |
| **Determinism** | Enforced (same input → same output) |

## What the Agent Must Do

Write Python code defining:

```python
def forecast(history_sales, history_promos, history_holidays, 
             history_leak_proxy, future_promos, future_holidays) -> list[float]:
    # Return 14 non-negative predictions
```

**Key challenge**: Model weekly/annual seasonality and promo/holiday effects well enough to beat a strong baseline. The `leak_proxy` feature is a trap - using it fails the leakage test.

## Setup & Run

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/RL agent final.git
cd RL agent final

# Set API key
# Windows:
set ANTHROPIC_API_KEY=your_key
# Linux/Mac:
export ANTHROPIC_API_KEY=your_key

# Install dependencies
pip install anthropic

# Run full evaluation (10 seeds)
python main.py

# Or run single test (saves API quota)
python main.py --single
```

## Expected Pass Rate: 10-40%

**Multiple failure modes (per rubric requirements):**
- Using `leak_proxy` → fails dual counterfactual test
- Ignoring annual seasonality → insufficient improvement
- Not handling promo/holiday effects → insufficient improvement
- Simple averages → ties baseline, doesn't beat by 12%
- Non-deterministic output → fails determinism check
- Predictions exceeding sanity bounds → fails bounds check
- Complex slow code → timeout

## Tuning Pass Rate

Edit config constants in `main.py`:

```python
IMPROVEMENT_REQ = 0.12  # Lower = easier (try 0.08 or 0.10)
LEAK_SENS_MAX = 0.02    # Higher = more lenient on leakage
```


## Files

- `main.py` - Complete task with single/full run modes
- `pyproject.toml` - Dependencies
- `README.md` - This file
