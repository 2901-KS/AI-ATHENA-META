"""
Baseline inference script for the SQL Query Optimizer OpenEnv environment.

Uses the OpenAI API (gpt-4o-mini by default) to run an agent against all 3 tasks
and produce reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline_inference.py [--model gpt-4o-mini] [--seed 42] [--verbose]

Output:
    Prints scores for each task and the aggregate weighted score.
    Saves results to baseline_results.json.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SQLQueryOptimizerEnv
from env.models import Action
from graders import run_all_graders


# ──────────────────────────────────────────────
# Agent prompt templates
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL query optimizer. Your goal is to rewrite slow SQL queries
to be as fast as possible while returning IDENTICAL results.

You will receive:
1. The current slow SQL query
2. Database schema information (tables, columns, row counts, available indexes)
3. Performance metrics (execution time, rows scanned, whether indexes are used)
4. Feedback on your previous attempt

Your response must be a JSON object with these fields:
{
  "optimized_query": "<your optimized SQL SELECT statement>",
  "reasoning": "<brief explanation of what you changed and why>",
  "declare_done": <true if you are satisfied with the optimization, false to continue>
}

Key optimization strategies:
- Replace correlated subqueries with JOINs or CTEs
- Add appropriate indexes (mention them in reasoning; the env can create them)
- Reorder joins to filter early
- Use window functions instead of self-joins for analytics
- Avoid SELECT * in subqueries
- Pre-aggregate before joining

IMPORTANT: Your optimized_query must be a valid SQLite SELECT statement.
Only return the JSON object, nothing else."""


def build_user_message(obs) -> str:
    """Build the user message for the agent from an observation."""
    schema_summary = []
    for table, info in obs.schema_info.items():
        if table.startswith("_"):
            continue
        cols = ", ".join(c["name"] for c in info["columns"])
        schema_summary.append(f"  {table} ({info['row_count']} rows): {cols}")
        if info["indexes"]:
            schema_summary.append(f"    indexes: {', '.join(info['indexes'])}")

    available_indexes = obs.schema_info.get("_available_indexes", [])
    active_indexes = obs.schema_info.get("_active_indexes", [])

    return f"""TASK ({obs.difficulty}): {obs.task_description}

CURRENT QUERY:
{obs.current_query}

PERFORMANCE:
- Execution time: {obs.current_metrics.execution_time_ms:.2f}ms
- Baseline time:  {obs.baseline_metrics.execution_time_ms:.2f}ms
- Speedup so far: {obs.baseline_metrics.execution_time_ms / max(obs.current_metrics.execution_time_ms, 0.1):.2f}x
- Rows scanned:   {obs.current_metrics.rows_scanned}
- Index used:     {obs.current_metrics.used_index}
- Result correct: {obs.current_metrics.result_hash == obs.baseline_metrics.result_hash}

DATABASE SCHEMA:
{chr(10).join(schema_summary)}

AVAILABLE INDEXES (not yet created): {', '.join(set(available_indexes) - set(active_indexes))}
ACTIVE INDEXES: {', '.join(active_indexes) if active_indexes else 'none'}

FEEDBACK FROM LAST ACTION: {obs.last_action_feedback}

Step {obs.step_number}/{obs.max_steps} | Cumulative reward: {obs.cumulative_reward:.4f}"""


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def run_agent_on_task(
    difficulty: str,
    client: OpenAI,
    model: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode of the agent on a given difficulty task."""
    env = SQLQueryOptimizerEnv(difficulty=difficulty, seed=seed)
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {obs.task_id} | Baseline: {obs.baseline_metrics.execution_time_ms:.2f}ms")
        print(f"{'='*60}")

    best_query = obs.current_query
    best_speedup = 1.0
    total_reward = 0.0
    hints_used = 0
    messages = []

    for step in range(obs.max_steps):
        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # Keep context window manageable
        if len(messages) > 10:
            messages = messages[-10:]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
        except Exception as e:
            if verbose:
                print(f"Step {step+1}: API error: {e}")
            break

        optimized_query = parsed.get("optimized_query", obs.current_query)
        reasoning = parsed.get("reasoning", "")
        declare_done = parsed.get("declare_done", False)

        if verbose:
            speedup = obs.baseline_metrics.execution_time_ms / max(obs.current_metrics.execution_time_ms, 0.1)
            print(f"Step {step+1}: {reasoning[:80]}... | Current speedup: {speedup:.2f}x")

        action = Action(
            optimized_query=optimized_query,
            request_hint=False,
            declare_done=declare_done,
            reasoning=reasoning,
        )

        obs, reward, done, info = env.step(action)
        total_reward += reward.total

        messages.append({"role": "assistant", "content": raw})

        if reward.speedup_ratio > best_speedup and reward.is_correct:
            best_speedup = reward.speedup_ratio
            best_query = optimized_query

        if verbose:
            print(f"  Reward: {reward.total:.4f} | Speedup: {reward.speedup_ratio:.2f}x | Correct: {reward.is_correct}")

        if done:
            break

        time.sleep(0.2)  # Rate limit courtesy

    return {
        "difficulty": difficulty,
        "best_query": best_query,
        "best_speedup": best_speedup,
        "hints_used": hints_used,
        "total_reward": round(total_reward, 4),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for SQL Query Optimizer OpenEnv")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step details")
    parser.add_argument("--output", default="baseline_results.json", help="Output file for results")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Running baseline inference with model={args.model}, seed={args.seed}")
    print("This will run 3 episodes (easy, medium, hard)...\n")

    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        print(f"Running {difficulty} task...")
        result = run_agent_on_task(
            difficulty=difficulty,
            client=client,
            model=args.model,
            seed=args.seed,
            verbose=args.verbose,
        )
        results[difficulty] = result
        print(f"  Best speedup: {result['best_speedup']:.2f}x | Total reward: {result['total_reward']:.4f}")

    # Run graders
    print("\nRunning graders...")
    grade_results = run_all_graders(
        easy_query=results["easy"]["best_query"],
        medium_query=results["medium"]["best_query"],
        hard_query=results["hard"]["best_query"],
        hints_easy=results["easy"]["hints_used"],
        hints_medium=results["medium"]["hints_used"],
        hints_hard=results["hard"]["hints_used"],
        seed=args.seed,
    )

    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Easy   score: {grade_results['easy']['score']:.4f}  (passed: {grade_results['easy']['passed']})")
    print(f"Medium score: {grade_results['medium']['score']:.4f}  (passed: {grade_results['medium']['passed']})")
    print(f"Hard   score: {grade_results['hard']['score']:.4f}  (passed: {grade_results['hard']['passed']})")
    print(f"{'='*60}")
    print(f"AGGREGATE SCORE: {grade_results['aggregate_score']:.4f}")
    print("="*60)

    # Save results
    output = {
        "model": args.model,
        "seed": args.seed,
        "episode_results": results,
        "grade_results": grade_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
