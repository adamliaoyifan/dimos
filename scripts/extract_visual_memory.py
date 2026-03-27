#!/usr/bin/env python3
"""Extract keyframe images from a visual_memory.pkl file.

Usage:
    python scripts/extract_visual_memory.py [PKL_PATH] [OUTPUT_DIR]

Defaults:
    PKL_PATH   = assets/output/memory/spatial_memory/visual_memory.pkl
    OUTPUT_DIR = assets/output/memory/spatial_memory/frames/
"""

import base64
import os
import pickle
import sys


def extract(pkl_path: str, output_dir: str) -> None:
    with open(pkl_path, "rb") as f:
        data: dict[str, str] = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for i, (key, b64_str) in enumerate(sorted(data.items())):
        img_bytes = base64.b64decode(b64_str)
        filename = f"{key}.jpg"
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as out:
            out.write(img_bytes)

    print(f"Extracted {len(data)} frames to {output_dir}")


if __name__ == "__main__":
    default_pkl = "assets/output/memory/spatial_memory/visual_memory.pkl"
    default_out = "assets/output/memory/spatial_memory/frames"

    pkl_path = sys.argv[1] if len(sys.argv) > 1 else default_pkl
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_out

    extract(pkl_path, output_dir)
