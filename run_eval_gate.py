import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="One-command evaluation runner with quality gate and failed-case suggestions."
    )
    parser.add_argument("--mode", choices=["api", "direct", "direct-mocked"], default="direct-mocked")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--cases", default="eval_questions.json")
    parser.add_argument("--min-pass-rate", type=float, default=95.0)
    parser.add_argument("--target-cases", type=int, default=60)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "auto_eval.py",
        "--mode", args.mode,
        "--api-url", args.api_url,
        "--cases", args.cases,
        "--min-pass-rate", str(args.min_pass_rate),
        "--target-cases", str(args.target_cases),
        "--failed-only"
    ]

    completed = subprocess.run(cmd)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
