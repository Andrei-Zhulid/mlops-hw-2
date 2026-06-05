"""Run judge_cases.json against the live judge and report pass/fail per case."""

import json
from pathlib import Path

from src.judge import Verdict, judge

CASES_FILE = Path(__file__).parent / "judge_cases.json"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def run() -> None:
    cases = json.loads(CASES_FILE.read_text())
    passed = 0
    failed = 0

    print(f"\nRunning {len(cases)} judge test cases...\n")
    print(f"{'ID':<45} {'EXPECTED':<22} {'GOT':<22} {'RESULT'}")
    print("-" * 105)

    for case in cases:
        result = judge(case["user_message"], case["assistant_response"])
        expected = Verdict(case["expected_verdict"])
        ok = result.verdict == expected
        status = PASS if ok else FAIL

        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{case['id']:<45} {expected.value:<22} {result.verdict.value:<22} {status}")
        if not ok:
            print(f"  edge_case: {case['edge_case']}")

    total = passed + failed
    print(f"\nResult: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print()


if __name__ == "__main__":
    run()