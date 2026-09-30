#!/usr/bin/env python3
"""
Plot a per-class IoU table image from a semantic_iou_raw.npy or
temporal_iou_raw.npy file.

Both files are 0-d object arrays wrapping a dict:
    {class_id (int): np.ndarray of per-frame (or per-frame-pair) IoU values
                     (float32, may contain NaN)}

For each Cityscapes class we compute summary statistics (mean / std / min / max /
median / valid-frame count) and render them as a single image of a table.
Use --mean to render a compact, easy-to-read table containing only the
per-class mean IoU.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CITYSCAPES_CLASSES = [
    {"id": 0,  "name": "road"},
    {"id": 1,  "name": "sidewalk"},
    {"id": 2,  "name": "building"},
    {"id": 3,  "name": "wall"},
    {"id": 4,  "name": "fence"},
    {"id": 5,  "name": "pole"},
    {"id": 6,  "name": "traffic light"},
    {"id": 7,  "name": "traffic sign"},
    {"id": 8,  "name": "vegetation"},
    {"id": 9,  "name": "terrain"},
    {"id": 10, "name": "sky"},
    {"id": 11, "name": "person"},
    {"id": 12, "name": "rider"},
    {"id": 13, "name": "car"},
    {"id": 14, "name": "truck"},
    {"id": 15, "name": "bus"},
    {"id": 16, "name": "train"},
    {"id": 17, "name": "motorcycle"},
    {"id": 18, "name": "bicycle"},
]

FILE_KIND_BY_NAME = {
    "semantic_iou_raw.npy": "semantic",
    "temporal_iou_raw.npy": "temporal",
}

KIND_LABELS = {
    "semantic": "Semantic IoU (per frame)",
    "temporal": "Temporal IoU (per frame pair, warped t → t+1)",
}


def _fmt(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "—"



def _row_color(mean_iou: float) -> str:
    if not np.isfinite(mean_iou):
        return "#f2f2f2"
    if mean_iou >= 0.75:
        return "#d8f3dc"
    if mean_iou >= 0.5:
        return "#fff3bf"
    if mean_iou >= 0.25:
        return "#ffd6a5"
    return "#ffadad"


def detect_kind(path: Path) -> str:
    kind = FILE_KIND_BY_NAME.get(path.name)
    if kind is not None:
        return kind
    # Fallback: look for a sibling file with a known name
    sibling = path.parent / "temporal_iou_raw.npy"
    if sibling.exists() and sibling.resolve() == path.resolve():
        return "temporal"
    return "semantic"


def build_table_data(raw: dict, mean_only: bool) -> tuple[list[list[str]], list[str]]:
    name_by_id = {c["id"]: c["name"] for c in CITYSCAPES_CLASSES}

    rows: list[list[str]] = []
    row_colors: list[str] = []
    for cid in sorted(raw.keys()):
        vals = np.asarray(raw[cid], dtype=np.float32)
        valid = vals[~np.isnan(vals)]

        if valid.size == 0:
            mean_iou = float("nan")
            n_valid = 0
        else:
            mean_iou = float(valid.mean())
            n_valid = int(valid.size)

        if mean_only:
            rows.append([
                str(cid),
                name_by_id.get(cid, f"class_{cid}"),
                _fmt(mean_iou),
            ])
        else:
            if valid.size == 0:
                std_iou = min_iou = max_iou = med_iou = float("nan")
            else:
                std_iou = float(valid.std())
                min_iou = float(valid.min())
                max_iou = float(valid.max())
                med_iou = float(np.median(valid))
            rows.append([
                str(cid),
                name_by_id.get(cid, f"class_{cid}"),
                _fmt(mean_iou),
                _fmt(std_iou),
                _fmt(min_iou),
                _fmt(max_iou),
                _fmt(med_iou),
                f"{n_valid}/{len(vals)}",
            ])

        row_colors.append(_row_color(mean_iou))

    return rows, row_colors


def render_table_image(
    rows: list[list[str]],
    row_colors: list[str],
    title: str,
    output: Path,
    dpi: int,
    mean_only: bool,
) -> None:
    headers = (
        ["ID", "Class", "Mean IoU"]
        if mean_only
        else ["ID", "Class", "Mean", "Std", "Min", "Max", "Median", "Valid frames"]
    )
    n_rows = len(rows) + 1  # +1 for header
    n_cols = len(headers)

    fig_w = 6.5 if mean_only else 11.0
    fig_h = max(2.5, 0.42 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12 if mean_only else 11)
    table.scale(1.0, 1.4)

    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_facecolor("#343a40")
        cell.set_text_props(color="white", weight="bold")

    for r, color in enumerate(row_colors, start=1):
        for col in range(n_cols):
            table[(r, col)].set_facecolor(color)

    # Emphasize the "Mean" column
    mean_col = headers.index("Mean IoU" if mean_only else "Mean")
    for r in range(1, n_rows):
        table[(r, mean_col)].set_text_props(weight="bold", size=13 if mean_only else 11)

    ax.set_title(title, fontsize=14, weight="bold", pad=12)

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a per-class IoU summary table image from "
            "semantic_iou_raw.npy or temporal_iou_raw.npy."
        )
    )
    parser.add_argument("npy_file", type=Path, help="Path to *_iou_raw.npy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <npy_file_parent>/per_class_iou_table.png)",
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Show only the per-class mean IoU column (compact view).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    raw_obj = np.load(args.npy_file, allow_pickle=True)
    raw = raw_obj.item()
    if not isinstance(raw, dict):
        raise ValueError(f"Expected *_iou_raw.npy to contain a dict, got {type(raw)}")

    kind = detect_kind(args.npy_file)
    suffix = "_mean" if args.mean else ""
    default_name = f"per_class_{kind}_iou_table{suffix}.png"
    output = args.output or args.npy_file.parent / default_name

    rows, row_colors = build_table_data(raw, mean_only=args.mean)
    title = f"Per-class {KIND_LABELS[kind]} — {args.npy_file.parent.name}"
    render_table_image(rows, row_colors, title, output, args.dpi, mean_only=args.mean)

    valid_means = [
        float(np.nanmean(np.asarray(raw[c["id"]], dtype=np.float32)))
        for c in CITYSCAPES_CLASSES
        if c["id"] in raw and np.any(~np.isnan(np.asarray(raw[c["id"]], dtype=np.float32)))
    ]
    if valid_means:
        mIoU = float(np.mean(valid_means))
        print(f"mIoU (mean of per-class means over {len(valid_means)} classes): {mIoU:.3f}")

    print(f"Saved per-class IoU table to {output}")


if __name__ == "__main__":
    main()
