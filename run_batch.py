"""feed generate.py every prompt in prompts.txt, then walk away.

extra flags pass straight through, e.g.:
    python run_batch.py prompts.txt --iterations 400
"""
import subprocess
import sys


def main():
    args = sys.argv[1:]
    path = args.pop(0) if args and not args[0].startswith("-") else "prompts.txt"
    with open(path) as f:
        prompts = [line.strip() for line in f
                   if line.strip() and not line.startswith("#")]
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt}")
        subprocess.run([sys.executable, "generate.py", prompt] + args, check=True)


if __name__ == "__main__":
    main()
