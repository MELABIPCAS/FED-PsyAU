"""校验特征目录是否符合训练所需规格。"""

import argparse
import json
import os

import cv2

AU_NAMES = {1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17}
SOURCES = ["first.png", "apex.png", "apex_pre.png", "apex_next.png"]
FLOWS = [("flow28.png", 28), ("flow128.png", 128),
         ("flow_pre28.png", 28), ("flow_pre128.png", 128),
         ("flow_next28.png", 28), ("flow_next128.png", 128)]


def check_sample(folder):
    errors, warnings = [], []
    for name in SOURCES + [name for name, _ in FLOWS] + ["landmarks_v2.json", "au.csv"]:
        if not os.path.exists(os.path.join(folder, name)):
            errors.append(f"缺少 {name}")

    for name, size in FLOWS:
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            continue
        if image.shape != (size, size, 3):
            errors.append(f"{name} 形状 {image.shape}，应为 {(size, size, 3)}")

    landmark_path = os.path.join(folder, "landmarks_v2.json")
    if os.path.exists(landmark_path):
        with open(landmark_path) as f:
            points = json.load(f)
        if len(points) != 161 or len(points[0]) != 2:
            errors.append(f"landmarks_v2.json 为 {len(points)} 点")

    au_path = os.path.join(folder, "au.csv")
    if os.path.exists(au_path):
        with open(au_path) as f:
            rows = [row.strip() for row in f.read().splitlines() if row.strip()]
        if not rows or rows[0] != "au":
            errors.append("au.csv 缺少 au 表头")
        for value in rows[1:]:
            if not value.isdigit() or int(value) not in AU_NAMES:
                warnings.append(f"au.csv 含非 12 类 AU 编码 {value}")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="特征规格校验")
    parser.add_argument("root", help="dfme_augment 根目录或单个样本目录")
    args = parser.parse_args()

    samples, passed, all_errors, all_warnings = 0, 0, [], []
    for folder, _, files in os.walk(args.root):
        if "apex.png" not in files:
            continue
        samples += 1
        errors, warnings = check_sample(folder)
        all_errors.extend(f"{folder}: {err}" for err in errors)
        all_warnings.extend(f"{folder}: {warn}" for warn in warnings)
        passed += not errors

    print(f"样本 {samples}，通过 {passed}，错误 {len(all_errors)}，警告 {len(all_warnings)}")
    for line in all_errors[:10]:
        print("  E", line)
    for line in all_warnings[:5]:
        print("  W", line)


if __name__ == "__main__":
    main()
