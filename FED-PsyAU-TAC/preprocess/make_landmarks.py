"""在已对齐的 128x128 帧上提取 161 个 landmark，写入 landmarks_v2.json。"""

import argparse
import json
import os

import cv2
import dlib

EYEBROW_LEFT = (17, 22)
EYEBROW_RIGHT = (22, 27)
EYE_LEFT = (36, 42)
EYE_RIGHT = (42, 48)
NOSE = (27, 36)
LOWER_NOSE = (31, 36)
MOUTH = (48, 68)
CHIN = (5, 12)


def midpoint(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def detect_landmarks(image, predictor):
    """68 个 dlib 点 + 7 个几何中点，按上脸 / 鼻嘴 / 下脸三组拼成 161 点。"""
    shape = predictor(image, dlib.rectangle(0, 0, image.shape[1], image.shape[0]))
    points = [(p.x, p.y) for p in shape.parts()]

    eyebrow_left = points[EYEBROW_LEFT[0]:EYEBROW_LEFT[1]]
    eyebrow_right = points[EYEBROW_RIGHT[0]:EYEBROW_RIGHT[1]]
    eye_left = points[EYE_LEFT[0]:EYE_LEFT[1]]
    eye_right = points[EYE_RIGHT[0]:EYE_RIGHT[1]]
    nose = points[NOSE[0]:NOSE[1]]
    lower_nose = points[LOWER_NOSE[0]:LOWER_NOSE[1]]
    mouth = points[MOUTH[0]:MOUTH[1]]
    chin = points[CHIN[0]:CHIN[1]]

    # 7 个补点：眉间中点，以及被丢弃的外颊点与眼角、鼻翼、嘴角之间的中点
    eyebrow_mid = midpoint(points[21], points[22])
    left_cheek = [midpoint(points[1], points[36]), midpoint(points[2], points[31]),
                  midpoint(points[3], points[48])]
    right_cheek = [midpoint(points[15], points[45]), midpoint(points[14], points[35]),
                   midpoint(points[13], points[54])]

    upper = eyebrow_left + eyebrow_right + eye_left + eye_right + [eyebrow_mid]
    return (upper
            + eyebrow_left + eyebrow_right + eye_left + eye_right + mouth
            + left_cheek + right_cheek
            + eyebrow_left + eyebrow_right + eye_left + eye_right + [eyebrow_mid]
            + nose + mouth + left_cheek + right_cheek + lower_nose + mouth + chin)


def main():
    parser = argparse.ArgumentParser(description="DFME 161 点 landmark 生成")
    parser.add_argument("root", help="dfme_augment 根目录")
    parser.add_argument("--predictor", default="../weight/shape_predictor_68_face_landmarks.dat",
                        help="dlib 68 点模型路径")
    args = parser.parse_args()

    predictor = dlib.shape_predictor(args.predictor)
    for folder, _, files in os.walk(args.root):
        if "apex.png" not in files:
            continue
        image = cv2.resize(dlib.load_rgb_image(os.path.join(folder, "apex.png")), (128, 128))
        points = detect_landmarks(image, predictor)
        with open(os.path.join(folder, "landmarks_v2.json"), "w") as f:
            json.dump(points, f)
        print(folder)


if __name__ == "__main__":
    main()
