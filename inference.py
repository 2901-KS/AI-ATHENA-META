"""
Inference Script — SQL Debug Environment
==========================================
MANDATORY VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "hf_placeholder"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "sql-debug-env"

TASKS = ["easy_syntax_fix", "medium_logic_fix", "hard_optimization_fix"]
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.7

# ─── Logging helpers (MANDATORY FORMAT) ──────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Escape newlines in action for single-line output
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    # Truncate if very long
    if len(action_clean) > 200:
        action_clean = action_clean[:197] + "..."
    error_val = error.replace("\n", " ") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─── Environment HTTP Client ──────────────────────────────────────────────────

def env_reset(task_name: str, session_id: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_step(corrected_sql: str, session_id: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"corrected_sql": corrected_sql, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL engineer. Your job is to fix broken SQL queries.

You will be given:
1. A broken SQL query
2. The error message it produces
3. The database schema (tables and columns)
4. A description of what the query should return

Output ONLY the corrected SQL query — no explanations, no markdown, no code blocks.
Just the raw SQL statement ending with a semicolon.

Rules:
- Use only columns that exist in the schema
- Match the expected result description exactly
- Fix ALL errors, not just the first one
- Do not add extra columns or change the core logic unless needed for correctness
""").strip()

def build_user_prompt(obs: dict, attempt: int, last_result: Optional[str]) -> str:
    schema_str = ""
    for tbl in obs.get("db_schema", []):
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in tbl["columns"])
        schema_str += f"\nTable: {tbl['table_name']}({cols})"
        if tbl.get("sample_rows"):
            schema_str += f"\n  Sample: {tbl['sample_rows'][:2]}"

    prompt = textwrap.dedent(f"""
BROKEN SQL (attempt {attempt}):
{obs['broken_sql']}

ERROR MESSAGE:
{obs['error_message']}

DATABASE SCHEMA:{schema_str}

EXPECTED BEHAVIOR:
{obs['expected_result_description']}
""").strip()

    if last_result and attempt > 1:
        prompt += f"\n\nLAST ATTEMPT RESULT: {last_result}"
        prompt += f"\nPartial score so far: {obs.get('partial_score', 0):.2f}"

    prompt += "\n\nWrite the corrected SQL query:"
    return prompt


def get_model_sql(client: OpenAI, obs: dict, attempt: int, last_result: Optional[str]) -> str:
    user_prompt = build_user_prompt(obs, attempt, last_result)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code blocks if model wraps in them
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last ``` lines
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        return text if text else "SELECT 1;"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;"


# ─── Run one task episode ─────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run a full episode for a given task.
    Emits [START], [STEP]..., [END] to stdout.
    Returns the final normalized score in [0, 1].
    """
    session_id = f"inference-{task_name}"
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset episode
        reset_data = env_reset(task_name, session_id)
        obs = reset_data["observation"]
        last_result = None

        for step in range(1, MAX_STEPS + 1):
            # Get model's corrected SQL
            corrected_sql = get_model_sql(client, obs, step, last_result)

            # Submit to environment
            step_data = env_step(corrected_sql, session_id)
            reward = step_data["reward"]
            done = step_data["done"]
            info = step_data.get("info", {})
            error = info.get("feedback") if not info.get("syntax_valid") else None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=corrected_sql,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                obs = step_data["observation"]
                last_result = obs.get("last_execution_result")
                break

            obs = step_data["observation"]
            last_result = obs.get("last_execution_result")

        # Score = best reward achieved in any step (agent gets credit for best attempt)
        # Also factor in whether it converged (final step was correct)
        if rewards:
            best_reward = max(rewards)
            final_reward = rewards[-1]
            # Weighted: 60% best, 40% final (encourages convergence)
            score = 0.6 * best_reward + 0.4 * final_reward
        else:
            score = 0.001

        score = round(min(0.999, max(0.001, score)), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} failed: {exc}", flush=True)
        if not rewards:
            rewards = [0.001]

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    task_names = os.getenv("SQL_DEBUG_TASKS", ",".join(TASKS)).split(",")

    for task_name in task_names:
        task_name = task_name.strip()
        if not task_name:
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            score = run_task(client, task_name)
            all_scores.append(score)
            print(f"[INFO] Task {task_name} score: {score:.3f}", flush=True)
        except Exception as exc:
            print(f"[ERROR] Task {task_name} crashed: {exc}", flush=True)
            all_scores.append(0.001)

    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f"\n{'='*60}", flush=True)
        print(f"FINAL AVERAGE SCORE: {avg:.3f}", flush=True)
        print(f"Task scores: {dict(zip(task_names, all_scores))}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
