from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="filtered")
    parser.add_argument("--output-dir", default="final_dataset")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--accepted-review-statuses", default="approved,keep")
    return parser.parse_args()


def read_clip_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_decision(text: str | None) -> str:
    return (text or "").strip().lower()


def resolve_final_decision(row: dict, include_review: bool, accepted_statuses: set[str]) -> str:
    review_status = normalize_decision(row.get("review_status"))
    auto_decision = normalize_decision(row.get("auto_decision"))

    if review_status in accepted_statuses:
        return "keep"
    if review_status in {"reject", "rejected", "drop"}:
        return "reject"
    if auto_decision == "keep":
        return "keep"
    if auto_decision == "review" and include_review:
        return "keep"
    return "reject"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    ensure_parent(destination)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_statuses = {
        status.strip().lower()
        for status in args.accepted_review_statuses.split(",")
        if status.strip()
    }

    clip_csv_files = sorted(input_dir.glob("*/clip_scores.csv"))
    if not clip_csv_files:
        print(f"clip_scores.csv not found under {input_dir}", file=sys.stderr)
        return 2

    manifest: list[dict] = []
    summary: list[dict] = []

    for csv_path in clip_csv_files:
        rows = read_clip_rows(csv_path)
        style_slug = csv_path.parent.name
        kept_for_style = 0

        for row in rows:
            decision = resolve_final_decision(
                row,
                include_review=args.include_review,
                accepted_statuses=accepted_statuses,
            )
            if decision != "keep":
                continue

            source_path = Path(row["image_path"]).resolve()
            final_style = (row.get("final_style") or "").strip() or row.get("style_name") or style_slug
            target_dir = output_dir / final_style
            target_path = target_dir / source_path.name
            transfer_file(source_path, target_path, args.mode)

            kept_for_style += 1
            manifest.append(
                {
                    **row,
                    "resolved_decision": decision,
                    "dataset_style": final_style,
                    "final_path": str(target_path),
                }
            )

        summary.append(
            {
                "style_slug": style_slug,
                "source_csv": str(csv_path),
                "kept_count": kept_for_style,
            }
        )

    write_json(output_dir / "dataset_manifest.json", manifest)
    write_csv(output_dir / "dataset_manifest.csv", manifest)
    write_json(output_dir / "dataset_summary.json", summary)
    write_csv(output_dir / "dataset_summary.csv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
