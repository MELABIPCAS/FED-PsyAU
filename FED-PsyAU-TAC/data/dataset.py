"""DFME 数据读取：全局光流、65 个 ROI patch、landmark 坐标。"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# DFME 各 part 对应的被试编号范围
DFME_FOLDERS = {
    "USTC_MEv4_partA_sub0001_sub0015": range(1, 16),
    "USTC_MEv4_partB_sub0016_sub0110": range(16, 111),
    "USTC_MEv4_partC_sub0111_sub0300": range(111, 301),
    "USTC_MEv4_partD_sub0301_sub0671": range(301, 672),
}

# 从 161 个对齐 landmark 中取出的 65 个 ROI
ROI_INDICES = list(range(0, 23)) + list(range(94, 123)) + list(range(-13, 0))


def get_sub_folder_7_class(emotion):
    mapping = {
        "anger": 0, "contempt": 1, "disgust": 2, "fear": 3,
        "happiness": 4, "sadness": 5, "surprise": 6,
    }
    return mapping.get(emotion.lower())


def get_au_labels(au_path, label_mapping):
    columns = pd.read_csv(au_path).iloc[:, 0]
    au_vector = [0] * len(label_mapping)
    for au in columns:
        if pd.notnull(au) and isinstance(au, (int, float)) and int(au) in label_mapping:
            au_vector[label_mapping[int(au)]] = 1
    return torch.tensor(au_vector)


def crop_patch(flow_image, point):
    """以 landmark 为中心裁 5x5 patch。"""
    start_x, end_x = point[0] - 2, point[0] + 3
    start_y, end_y = point[1] - 2, point[1] + 3
    patch = flow_image[start_x:end_x, start_y:end_y]
    patch = cv2.resize(patch, (5, 5), interpolation=cv2.INTER_LINEAR)
    return transforms.ToTensor()(patch)


def read_landmarks(landmarks_path):
    with open(landmarks_path, "r") as f:
        return json.load(f)


class MEDataset(Dataset):
    """给定被试的样本：全局光流、65 个 patch、坐标、AU 标签、表情标签。"""

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, image_root: str,
                 subjects: list, triple_frames: bool = False):
        self.label_mapping = label_mapping
        self.samples = []
        subject_names = set(subjects)
        for _, row in data_info.iterrows():
            subject_folder, filename, emotion_category = row[0], row[1], row[6]
            if subject_folder not in subject_names:
                continue
            label = get_sub_folder_7_class(emotion_category)
            if label is None:
                continue

            sub_id = int(subject_folder.replace("sub", ""))
            target_folder = None
            for folder, sub_range in DFME_FOLDERS.items():
                if sub_id in sub_range:
                    if "partD" in folder:
                        target_folder = f"{folder}_{int(row['fps'])}fps"
                    else:
                        target_folder = folder
                    break
            folder_path = os.path.join(
                image_root, target_folder, "micro_static_aligned_croped", emotion_category, filename)

            frames = ["flow28.png", "flow128.png"]
            if triple_frames:
                frames = ["flow_pre28.png", "flow_pre128.png",
                          "flow28.png", "flow128.png",
                          "flow_next28.png", "flow_next128.png"]
                for global_name, patch_name in zip(frames[0::2], frames[1::2]):
                    self.samples.append((os.path.join(folder_path, global_name),
                                         os.path.join(folder_path, patch_name),
                                         os.path.join(folder_path, "au.csv"), label))
            else:
                self.samples.append((os.path.join(folder_path, frames[0]),
                                     os.path.join(folder_path, frames[1]),
                                     os.path.join(folder_path, "au.csv"), label))

        self.data = [self._load(*sample) for sample in self.samples]

    def _load(self, global_flow_path, patch_flow_path, au_path, label):
        global_flow = transforms.ToTensor()(cv2.imread(global_flow_path))
        patch_flow = cv2.imread(patch_flow_path)
        landmarks = torch.tensor(read_landmarks(
            os.path.join(os.path.dirname(global_flow_path), "landmarks_v2.json")), dtype=torch.float32)
        points = landmarks[ROI_INDICES]
        patches = torch.cat([crop_patch(patch_flow, point) for point in points], dim=0)
        return global_flow, patches, points, get_au_labels(au_path, self.label_mapping), label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global_flow, patches, points, au_labels, label = self.data[idx]
        return global_flow, patches, points, au_labels, torch.tensor(label)


class AugmentedDataset(Dataset):
    """用随机小角度旋转扩充少数类。"""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.rotate_right = transforms.RandomRotation(degrees=(0, 3))
        self.rotate_left = transforms.RandomRotation(degrees=(-3, 0))
        self.rotate_both = transforms.RandomRotation(degrees=(-3, 3))
        self.augmented_data = []
        for global_flow, patches, points, au_labels, label in original_dataset:
            if label in (3, 4, 6):
                self.augmented_data.append(
                    (self.rotate_both(global_flow), self.rotate_both(patches), points, au_labels, label))
            elif label in (0, 1, 5):
                self.augmented_data.append(
                    (self.rotate_right(global_flow), self.rotate_right(patches), points, au_labels, label))
                self.augmented_data.append(
                    (self.rotate_left(global_flow), self.rotate_left(patches), points, au_labels, label))

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        return self.augmented_data[idx - len(self.original_dataset)]


def split_subjects(subjects, test_ratio=0.3, seed=1):
    """按被试划分训练/测试集。"""
    subjects = np.array(sorted(subjects))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(subjects))
    num_test = int(len(subjects) * test_ratio)
    test = subjects[order[:num_test]]
    train = subjects[order[num_test:]]
    return list(train), list(test)
