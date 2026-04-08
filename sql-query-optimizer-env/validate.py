"""
OpenEnv Validation Script

Runs the automated pass/fail gate that judges use in Phase 1.
Tests all required OpenEnv spec compliance items:
  1. Environment can be instantiated for all 3 difficulties
  2. reset() returns a valid Observation
  3. step() returns (Observation, Reward, bool, dict)
  4. state() returns a dict with required keys
  5. Reward is in [-1.0, 1.0] range
  6. Graders return scores in [0.0, 1.0]
  7. Done flag terminates correctly
  8. Invalid SQL is handled gracefully (no crash)
  9. Hint system works
  10. Episode result is deterministic (same seed → same score)

Usage:
    python validate.py
    python validate.py --verbose
"""

import sys
import traceback
import argparse
from typing import List, Tuple

sys.path.insert(0, ".")

from env.environment import SQLQueryOptimizerEnv
from env.models import Action, Observation, Reward
from graders import EasyGrader, MediumGrader, HardGrader


PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: List[Tuple[str, str, str]] = []  # (test_name, status, detail)


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = PASS if condition else (WARN if warn_only else FAIL)
    results.append((name, status, detail))
    return condition


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ──────────────────────────────────────────────
# Test suites
# ──────────────────────────────────────────────

def test_instantiation(verbose: bool):
    section("1. Environment Instantiation")
    for diff in ["easy", "medium", "hard"]:
        try:
            env = SQLQueryOptimizerEnv(difficulty=diff, seed=42)
            check(f"Instantiate {diff}", True, f"SQLQueryOptimizerEnv(difficulty='{diff}')")
        except Exception as e:
            check(f"Instantiate {diff}", False, str(e))

    try:
        SQLQueryOptimizerEnv(difficulty="impossible")
        check("Reject invalid difficulty", False, "Should have raised ValueError")
    except ValueError:
        check("Reject invalid difficulty", True, "ValueError raised correctly")
    except Exception as e:
        check("Reject invalid difficulty", False, f"Wrong exception: {e}")


def test_reset(verbose: bool):
    section("2. reset() API")
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    try:
        obs = env.reset()
        check("reset() returns Observation", isinstance(obs, Observation), type(obs).__name__)
        check("observation.task_id set", bool(obs.task_id), obs.task_id)
        check("observation.current_query set", bool(obs.current_query), "")
        check("observation.baseline_metrics set", obs.baseline_metrics is not None, "")
        check("observation.schema_info has tables", len(obs.schema_info) > 0, f"{len(obs.schema_info)} keys")
        check("observation.step_number == 0", obs.step_number == 0, str(obs.step_number))
        check("observation.done == False", obs.done == False, str(obs.done))
        check("baseline execution_time_ms > 0", obs.baseline_metrics.execution_time_ms > 0,
              f"{obs.baseline_metrics.execution_time_ms:.2f}ms")
        if verbose:
            print(f"    Baseline time: {obs.baseline_metrics.execution_time_ms:.2f}ms")
            print(f"    Schema tables: {[k for k in obs.schema_info if not k.startswith('_')]}")
    except Exception as e:
        check("reset() did not crash", False, traceback.format_exc())


def test_step(verbose: bool):
    section("3. step() API")
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    obs = env.reset()

    # Step with a valid query
    action = Action(optimized_query=obs.current_query, declare_done=False)
    try:
        result = env.step(action)
        check("step() returns 4-tuple", len(result) == 4, str(len(result)))
        obs2, reward, done, info = result

        check("step() obs is Observation", isinstance(obs2, Observation), type(obs2).__name__)
        check("step() reward is Reward", isinstance(reward, Reward), type(reward).__name__)
        check("step() done is bool", isinstance(done, bool), str(type(done)))
        check("step() info is dict", isinstance(info, dict), str(type(info)))

        check("reward.total in [-1, 1]", -1.0 <= reward.total <= 1.0,
              f"reward.total = {reward.total}")
        check("reward.is_correct is bool", isinstance(reward.is_correct, bool), "")
        check("reward.speedup_ratio > 0", reward.speedup_ratio > 0, f"{reward.speedup_ratio}")
        check("obs.step_number incremented", obs2.step_number == 1, str(obs2.step_number))

        if verbose:
            print(f"    Reward: {reward.total:.4f} | Explanation: {reward.explanation[:60]}...")
    except Exception as e:
        check("step() did not crash", False, traceback.format_exc())

    # Step with invalid SQL
    bad_action = Action(optimized_query="THIS IS NOT SQL", declare_done=False)
    try:
        obs3, reward3, done3, info3 = env.step(bad_action)
        check("Invalid SQL handled gracefully", True, "No crash")
        check("Invalid SQL gives negative reward", reward3.total < 0, f"reward={reward3.total}")
        check("Invalid SQL: last_action_valid = False", not obs3.last_action_valid, "")
    except Exception as e:
        check("Invalid SQL handled gracefully", False, traceback.format_exc())


def test_done(verbose: bool):
    section("4. Episode Termination")
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    obs = env.reset()

    # declare_done=True should terminate immediately
    action = Action(optimized_query=obs.current_query, declare_done=True)
    obs2, reward, done, info = env.step(action)
    check("declare_done=True ends episode", done == True, f"done={done}")

    # Step after done should raise
    try:
        env.step(action)
        check("Step after done raises RuntimeError", False, "No exception raised")
    except RuntimeError:
        check("Step after done raises RuntimeError", True, "RuntimeError raised correctly")

    # Max steps termination
    env2 = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    env2.reset()
    env2._task.max_steps = 2  # Force short episode
    env2._step_num = 1
    action_nodone = Action(optimized_query=obs.current_query, declare_done=False)
    _, _, done2, _ = env2.step(action_nodone)
    check("Max steps ends episode", done2 == True, f"done={done2}")


def test_state(verbose: bool):
    section("5. state() API")
    env = SQLQueryOptimizerEnv(difficulty="medium", seed=42)
    env.reset()

    try:
        s = env.state()
        check("state() returns dict", isinstance(s, dict), type(s).__name__)
        required_keys = ["difficulty", "task_id", "step_number", "max_steps",
                         "current_query", "cumulative_reward", "done"]
        for key in required_keys:
            check(f"state() has '{key}'", key in s, f"keys: {list(s.keys())[:5]}")
        if verbose:
            print(f"    State keys: {list(s.keys())}")
    except Exception as e:
        check("state() did not crash", False, traceback.format_exc())


def test_graders(verbose: bool):
    section("6. Graders")
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    obs = env.reset()
    baseline_q = obs.current_query

    # Baseline query should score something (not crash)
    try:
        result = EasyGrader(seed=42).grade(baseline_q, hints_used=0)
        check("EasyGrader returns dict", isinstance(result, dict), "")
        check("EasyGrader score in [0,1]", 0.0 <= result["score"] <= 1.0,
              f"score={result['score']}")
        check("EasyGrader has 'passed' key", "passed" in result, "")
        check("EasyGrader has 'breakdown'", "breakdown" in result, "")
        if verbose:
            print(f"    Easy baseline score: {result['score']:.4f} | passed={result['passed']}")
    except Exception as e:
        check("EasyGrader did not crash", False, traceback.format_exc())

    try:
        env2 = SQLQueryOptimizerEnv(difficulty="medium", seed=42)
        obs2 = env2.reset()
        result2 = MediumGrader(seed=42).grade(obs2.current_query)
        check("MediumGrader score in [0,1]", 0.0 <= result2["score"] <= 1.0,
              f"score={result2['score']}")
    except Exception as e:
        check("MediumGrader did not crash", False, traceback.format_exc())

    try:
        env3 = SQLQueryOptimizerEnv(difficulty="hard", seed=42)
        obs3 = env3.reset()
        result3 = HardGrader(seed=42).grade(obs3.current_query)
        check("HardGrader score in [0,1]", 0.0 <= result3["score"] <= 1.0,
              f"score={result3['score']}")
    except Exception as e:
        check("HardGrader did not crash", False, traceback.format_exc())


def test_determinism(verbose: bool):
    section("7. Determinism & Reproducibility")
    # Same seed → same baseline time
    times = []
    for _ in range(2):
        env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
        obs = env.reset()
        times.append(obs.baseline_metrics.result_hash)

    check("Same seed → same result hash", times[0] == times[1],
          f"hashes: {times[0][:8]}... == {times[1][:8]}...")

    # Different seed → same structure (both valid)
    env_a = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    env_b = SQLQueryOptimizerEnv(difficulty="easy", seed=99)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    check("Different seeds produce same schema", obs_a.task_id == obs_b.task_id, "")


def test_hints(verbose: bool):
    section("8. Hint System")
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    obs = env.reset()

    action = Action(optimized_query=obs.current_query, request_hint=True, declare_done=False)
    obs2, reward, done, info = env.step(action)

    check("Hint request returns hint text", obs2.hint is not None and len(obs2.hint) > 0,
          obs2.hint[:50] if obs2.hint else "None")
    check("Hint costs negative reward", reward.hint_penalty < 0,
          f"hint_penalty={reward.hint_penalty}")
    if verbose:
        print(f"    Hint text: {obs2.hint[:80]}...")


def test_all_difficulties(verbose: bool):
    section("9. All Difficulties Full Episode")
    for diff in ["easy", "medium", "hard"]:
        try:
            env = SQLQueryOptimizerEnv(difficulty=diff, seed=42)
            obs = env.reset()
            # Run one step and declare done
            action = Action(optimized_query=obs.current_query, declare_done=True)
            obs2, reward, done, info = env.step(action)
            check(f"{diff}: full episode completes", done, f"done={done}")
            check(f"{diff}: state() works post-episode",
                  isinstance(env.state(), dict), "")
        except Exception as e:
            check(f"{diff}: full episode completes", False, str(e)[:60])


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenEnv validation for SQL Query Optimizer")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  SQL Query Optimizer — OpenEnv Validation")
    print("=" * 55)

    test_instantiation(args.verbose)
    test_reset(args.verbose)
    test_step(args.verbose)
    test_done(args.verbose)
    test_state(args.verbose)
    test_graders(args.verbose)
    test_determinism(args.verbose)
    test_hints(args.verbose)
    test_all_difficulties(args.verbose)

    # Summary
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    warned = sum(1 for _, s, _ in results if s == WARN)

    for name, status, detail in results:
        detail_str = f"  → {detail}" if detail and args.verbose else ""
        print(f"  {status}  {name}{detail_str}")

    print(f"\n{'─'*55}")
    print(f"  Total: {len(results)} tests | {passed} passed | {failed} failed | {warned} warnings")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED — Environment is OpenEnv compliant!")
        sys.exit(0)
    else:
        print(f"\n  ❌ {failed} test(s) FAILED — Fix before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
