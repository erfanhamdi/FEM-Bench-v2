"""Run one-pass implementation or test generation."""

from fem_bench.cli import main


if __name__ == "__main__":
    raise SystemExit(main("onepass", "onepass_code.yaml"))
