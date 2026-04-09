from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def read_styles_file(path: Path) -> list[str]:
    styles: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        styles.append(line)
    return styles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles-file", required=True)
    parser.add_argument("--limit-per-style", type=int, default=50)
    parser.add_argument("--output-dir", default="downloads_batch")
    parser.add_argument("--providers", default="wikimedia")
    parser.add_argument("--per-query-limit", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-filesize-mb", type=int, default=25)
    parser.add_argument("--download-delay", type=float, default=1.2)
    parser.add_argument("--wikimedia-thumb-width", type=int, default=1600)
    parser.add_argument("--extra-query", action="append", default=[])
    parser.add_argument("--allow-nc", action="store_true")
    parser.add_argument("--allow-nd", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def build_collect_command(args: argparse.Namespace, style: str) -> list[str]:
    command = [
        sys.executable,
        "collect_free_images.py",
        style,
        "--limit",
        str(args.limit_per_style),
        "--output-dir",
        args.output_dir,
        "--providers",
        args.providers,
        "--per-query-limit",
        str(args.per_query_limit),
        "--max-pages",
        str(args.max_pages),
        "--min-width",
        str(args.min_width),
        "--min-height",
        str(args.min_height),
        "--timeout",
        str(args.timeout),
        "--max-filesize-mb",
        str(args.max_filesize_mb),
        "--download-delay",
        str(args.download_delay),
        "--wikimedia-thumb-width",
        str(args.wikimedia_thumb_width),
    ]

    if args.allow_nc:
        command.append("--allow-nc")
    if args.allow_nd:
        command.append("--allow-nd")
    if args.dry_run:
        command.append("--dry-run")
    for query in args.extra_query:
        command.extend(["--extra-query", query])
    return command


def write_summary(records: list[dict], output_dir: Path) -> None:
    json_path = output_dir / "batch_summary.json"
    csv_path = output_dir / "batch_summary.csv"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def main() -> int:
    args = parse_args()
    styles_file = Path(args.styles_file).resolve()
    if not styles_file.exists():
        print(f"Файл со стилями не найден: {styles_file}", file=sys.stderr)
        return 2

    styles = read_styles_file(styles_file)
    if not styles:
        print(f"В файле нет стилей: {styles_file}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for style in styles:
        command = build_collect_command(args, style)
        started_at = time.time()
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent)
        elapsed = round(time.time() - started_at, 2)

        record = {
            "style": style,
            "exit_code": result.returncode,
            "elapsed_seconds": elapsed,
            "dry_run": args.dry_run,
            "output_dir": str(output_dir),
        }
        summary.append(record)

        if result.returncode != 0:
            print(f"{style}: exit {result.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                write_summary(summary, output_dir)
                return result.returncode

    write_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
