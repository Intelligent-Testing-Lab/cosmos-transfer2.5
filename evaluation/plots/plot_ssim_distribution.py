#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-frame SSIM values and their distribution from a JSON file."
    )
    parser.add_argument("json_file", type=Path, help="Path to the JSON file containing ssim.values")
    parser.add_argument("--output", type=Path, default=Path("ssim_distribution.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    with args.json_file.open() as f:
        data = json.load(f)

    values = data["ssim"]["values"]
    
    values_np = np.asanyarray(values)
    print(f"Max SSIM: {np.argmax(values_np)} Value: {values_np[np.argmax(values_np)]}")
    print(f"Min SSIM: {np.argmin(values_np)} Value: {values_np[np.argmin(values_np)]}")
    
    
    df = pd.DataFrame({"frame": range(len(values)), "ssim": values})

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax_frames = plt.subplots(figsize=(13, 5), constrained_layout=True)

    sns.lineplot(data=df, x="frame", y="ssim", ax=ax_frames, color="steelblue", linewidth=1)
    ax_frames.scatter(df["frame"], df["ssim"], s=8, color="steelblue", alpha=0.45)
    ax_frames.axhline(df["ssim"].mean(), color="crimson", linestyle="--", linewidth=1.2,
                      label=f"Mean: {df['ssim'].mean():.3f}")
    ax_frames.axhline(df["ssim"].median(), color="darkorange", linestyle=":", linewidth=1.4,
                      label=f"Median: {df['ssim'].median():.3f}")
    ax_frames.set(title="SSIM Across Frames", xlabel="Frame index", ylabel="SSIM")
    ax_frames.legend()

    #sns.histplot(data=df, x="ssim", bins=30, kde=True, ax=ax_dist, color="steelblue", edgecolor="white")
    #ax_dist.axvline(df["ssim"].mean(), color="crimson", linestyle="--", linewidth=1.2)
    #ax_dist.axvline(df["ssim"].median(), color="darkorange", linestyle=":", linewidth=1.4)
    #ax_dist.set(title="Distribution of Per-Frame SSIM", xlabel="SSIM", ylabel="Frame count")

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot for {len(df)} frames to {args.output}")


if __name__ == "__main__":
    main()