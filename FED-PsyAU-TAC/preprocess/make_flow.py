"""从对齐好的关键帧生成光流图（3 通道：光应变、垂直分量、水平分量）。"""

import argparse
import os

import cv2
import numpy as np

# 统一以 onset 帧 first.png 为参考帧，分别与三帧配对
PAIRS = [
    ("first.png", "apex.png", "flow"),
    ("first.png", "apex_pre.png", "flow_pre"),
    ("first.png", "apex_next.png", "flow_next"),
]

# PNG 体积上限，超过则先降采样再放大回来
SIZE_LIMITS = {28: 2 * 1024, 128: 20 * 1024}


def read_gray(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)


def minmax(x):
    top, bottom = np.max(x), np.min(x)
    if top == bottom:
        return x * 0
    return (x - bottom) / (top - bottom)


def optical_flow(start, end):
    flow = cv2.optflow.DualTVL1OpticalFlow_create().calc(start, end, None)
    u = minmax(flow[:, :, 0]) * 255
    v = minmax(flow[:, :, 1]) * 255
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    strain = np.sqrt(ux ** 2 + vy ** 2 + 0.5 * (uy + vx) ** 2)
    return np.concatenate((strain[..., None], v[..., None], u[..., None]), axis=2)


def save(flow, size, path):
    limit = SIZE_LIMITS[size]
    for i in range(size, 3, -1):
        image = cv2.resize(cv2.resize(flow, (i, i), interpolation=cv2.INTER_AREA),
                           (size, size), interpolation=cv2.INTER_NEAREST)
        ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if len(buffer) < limit:
            with open(path, "wb") as f:
                f.write(buffer.tobytes())
            return


def process(folder):
    for start_name, end_name, prefix in PAIRS:
        flow = optical_flow(read_gray(os.path.join(folder, start_name)),
                            read_gray(os.path.join(folder, end_name)))
        for size in (28, 128):
            save(flow, size, os.path.join(folder, f"{prefix}{size}.png"))


def main():
    parser = argparse.ArgumentParser(description="DFME 光流特征生成")
    parser.add_argument("root", help="dfme_augment 根目录")
    args = parser.parse_args()
    for folder, _, files in os.walk(args.root):
        if "first.png" in files and "apex.png" in files:
            process(folder)
            print(folder)


if __name__ == "__main__":
    main()
