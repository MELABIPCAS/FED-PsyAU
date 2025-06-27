from typing import Union
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from utils.landmark import detect_landmarks, read_detect_faces, detect_landmarks_v2
import os

dfme_folders = {
    'USTC_MEv4_partA_sub0001_sub0015': range(1, 16),
    'USTC_MEv4_partB_sub0016_sub0110': range(16, 111),
    'USTC_MEv4_partC_sub0111_sub0300': range(111, 301),
    'USTC_MEv4_partD_sub0301_sub0671': range(301, 672),
}
def flow_postprocessing(unit):
    unit = cv2.resize(unit, (128, 128), interpolation=cv2.INTER_LINEAR)
    return unit

def get_patches(point: tuple):
    start_x = point[0] - 2
    end_x = point[0] + 3

    start_y = point[1] - 2
    end_y = point[1] + 3

    if start_x == end_x:
        if start_x >= 64:
            start_x = end_x - 5
        else:
            end_x = start_x + 5
    if start_y == end_y:
        if start_y >= 64:
            start_y = end_y - 5
        else:
            end_y = start_y + 5

    if start_x < 0:
        start_x = 0
        end_x = 5
    elif end_x > 128:
        end_x = 128
        start_x = 123

    if start_y < 0:
        start_y = 0
        end_y = 5
    elif end_y > 128:
        end_y = 128
        start_y = 123

    return start_x, end_x, start_y, end_y


def get_sub_folder(emotion):
    if emotion.lower() == 'happy' or emotion.lower() == 'happiness':
        return '1'
    elif emotion.lower() == 'surprise':
        return '2'
    elif emotion.lower() == 'others' or emotion.lower() == 'other':
        return None
    else:
        return '0'

def get_sub_folder_7_class(emotion):
    if emotion.lower() == 'anger':
        return '0'
    elif emotion.lower() == 'contempt':
        return '1'
    elif emotion.lower() == 'disgust':
        return '2'
    elif emotion.lower() == 'fear':
        return '3'
    elif emotion.lower() == 'happy' or emotion.lower() == 'happiness':
        return '4'
    elif emotion.lower() == 'sadness' or emotion.lower() == 'sad':
        return '5'
    elif emotion.lower() == 'surprise':
        return '6'
    elif emotion.lower() == 'others' or emotion.lower() == 'other':
        return None

def get_sub_folder_7_class_casme3(emotion):
    if emotion.lower() == 'happy':
        return '0'
    elif emotion.lower() == 'disgust':
        return '1'
    elif emotion.lower() == 'fear':
        return '2'
    elif emotion.lower() == 'anger':
        return '3'
    elif emotion.lower() == 'sad':
        return '4'
    elif emotion.lower() == 'surprise':
        return '5'
    elif emotion.lower() == 'others':
        return '6'


def get_non_empty_numbers_from_first_column(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        first_column = df.iloc[:, 0]
        non_empty_numbers = [value for value in first_column if pd.notnull(value) and isinstance(value, (int, float))]
        return non_empty_numbers
    except Exception as e:
        print(f"An error occurred: {e}")
        return []





class MEDataset_161_landmark_7class_emo(Dataset):

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, image_root: str, category: str, device: torch.device, train: bool, test_subject_list: list):
        self.image_root = image_root
        self.data_info = data_info
        self.test_subject_list = test_subject_list
        self.label_mapping = label_mapping
        self.category = category
        self.train = train
        self.device = device
        self.transforms = transforms.ToTensor()
        self.transforms_dfme_flow = transforms.Compose([
            transforms.Lambda(lambda img: to_pil_image(img) if isinstance(img, np.ndarray) else img),
            transforms.ToTensor(),
        ])
        self.flow_train = []
        self.au_train = []
        self.label_train = []
        self.flow_test = []
        self.au_test = []
        self.label_test = []
        if self.category == 'DFME':
            for index, row in self.data_info.iterrows():
                subject_folder = row[0]
                fileame_folder = row[1]
                emotion_category = row[6]
                sub_id = int(row[0].replace('sub', ''))
                sub_folder = get_sub_folder_7_class(emotion_category)

                if sub_folder is not None and subject_folder not in self.test_subject_list:
                    for folder, sub_range in dfme_folders.items():
                        if sub_id in sub_range:
                            if 'partD_sub0301_sub0671' in folder:
                                if row['fps'] == 500:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_500fps'
                                elif row['fps'] == 200:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_200fps'
                            else:
                                target_folder = folder
                            break
                    folder_path = os.path.join(self.image_root, target_folder, 'micro_static_aligned_croped', emotion_category, fileame_folder)
                    # folder_path = os.path.join(self.image_root, target_folder, 'micro', emotion_category, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_train.append(flow_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and subject_folder in self.test_subject_list:
                    for folder, sub_range in dfme_folders.items():
                        if sub_id in sub_range:
                            if 'partD_sub0301_sub0671' in folder:
                                if row['fps'] == 500:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_500fps'
                                elif row['fps'] == 200:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_200fps'
                            else:
                                target_folder = folder
                            break
                    folder_path = os.path.join(self.image_root, target_folder, 'micro_static_aligned_croped', emotion_category, fileame_folder)
                    # folder_path = os.path.join(self.image_root, target_folder, 'micro', emotion_category, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_test.append(flow_path)
                    self.au_test.append(au_path)
                    self.label_test.append(int(sub_folder))
        self.train_data = self._preload_data(self.flow_train, self.au_train) if self.train else []
        self.test_data = self._preload_data(self.flow_test, self.au_test) if not self.train else []

    def _preload_data(self, flow_paths, au_paths):
        data = []
        for flow_path, au_path in zip(flow_paths, au_paths):

            parent_directory = os.path.dirname(flow_path)
            global_optical_flow = cv2.imread(os.path.join(parent_directory, 'flow28.png'))
            # optical_flow = cv2.imread(flow_path)
            # optical_flow = flow_postprocessing(optical_flow)
            optical_flow = cv2.imread(os.path.join(parent_directory, 'flow128.png'))
            points = detect_landmarks_v2(flow_path)

            patches = []
            for point in points:
                start_x, end_x, start_y, end_y = get_patches(point)
                patch = optical_flow[start_x:end_x, start_y:end_y]
                patch = cv2.resize(patch, (5, 5), interpolation=cv2.INTER_LINEAR)
                patch = transforms.ToTensor()(patch)
                patches.append(patch)
            patches = torch.cat(patches, dim=0)

            au_list = get_non_empty_numbers_from_first_column(au_path)
            vector_length = len(self.label_mapping)
            au_vector = [0] * vector_length
            for au in au_list:
                if au in self.label_mapping:
                    index = self.label_mapping[au]
                    au_vector[index] = 1
            if self.category == 'DFME':
                flow = self.transforms_dfme_flow(global_optical_flow)
            data.append((flow, patches, torch.tensor(au_vector)))

        return data

    def __len__(self):
        if self.train:
            return len(self.flow_train)
        else:
            return len(self.flow_test)

    def __getitem__(self, idx: int):
        if self.train:
            optical_flow, patches, au_vector = self.train_data[idx]
            label = self.label_train[idx]
        else:
            optical_flow, patches, au_vector = self.test_data[idx]
            label = self.label_test[idx]
        return optical_flow, patches, au_vector, torch.tensor(label)

class MEDataset_161_landmark_7class_emo_triple_frames(Dataset):

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, image_root: str, category: str, device: torch.device, train: bool, test_subject_list: list):
        self.image_root = image_root
        self.data_info = data_info
        self.test_subject_list = test_subject_list
        self.label_mapping = label_mapping
        self.category = category
        self.train = train
        self.device = device
        self.transforms = transforms.ToTensor()
        self.transforms_dfme_flow = transforms.Compose([
            transforms.Lambda(lambda img: to_pil_image(img) if isinstance(img, np.ndarray) else img),
            transforms.ToTensor(),
        ])
        self.global_flow_train = []
        self.patch_flow_train = []
        self.au_train = []
        self.label_train = []
        self.global_flow_test = []
        self.patch_flow_test = []
        self.au_test = []
        self.label_test = []
        if self.category == 'DFME':
            for index, row in self.data_info.iterrows():
                subject_folder = row[0]
                fileame_folder = row[1]
                emotion_category = row[6]
                sub_id = int(row[0].replace('sub', ''))
                sub_folder = get_sub_folder_7_class(emotion_category)

                if sub_folder is not None and subject_folder not in self.test_subject_list:
                    for folder, sub_range in dfme_folders.items():
                        if sub_id in sub_range:
                            if 'partD_sub0301_sub0671' in folder:
                                if row['fps'] == 500:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_500fps'
                                elif row['fps'] == 200:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_200fps'
                            else:
                                target_folder = folder
                            break
                    folder_path = os.path.join(self.image_root, target_folder, 'micro_static_aligned_croped', emotion_category, fileame_folder)
                    # folder_path = os.path.join(self.image_root, target_folder, 'micro', emotion_category, fileame_folder)
                    global_flow_path = os.path.join(folder_path, 'flow28.png')
                    global_flow_pre_path = os.path.join(folder_path, 'flow_pre28.png')
                    global_flow_next_path = os.path.join(folder_path, 'flow_next28.png')
                    patch_flow_path = os.path.join(folder_path, 'flow128.png')
                    patch_flow_pre_path = os.path.join(folder_path, 'flow_pre128.png')
                    patch_flow_next_path = os.path.join(folder_path, 'flow_next128.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.global_flow_train.append(global_flow_path)
                    self.patch_flow_train.append(patch_flow_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                    self.global_flow_train.append(global_flow_pre_path)
                    self.patch_flow_train.append(patch_flow_pre_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                    self.global_flow_train.append(global_flow_next_path)
                    self.patch_flow_train.append(patch_flow_next_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and subject_folder in self.test_subject_list:
                    for folder, sub_range in dfme_folders.items():
                        if sub_id in sub_range:
                            if 'partD_sub0301_sub0671' in folder:
                                if row['fps'] == 500:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_500fps'
                                elif row['fps'] == 200:
                                    target_folder = 'USTC_MEv4_partD_sub0301_sub0671_200fps'
                            else:
                                target_folder = folder
                            break
                    folder_path = os.path.join(self.image_root, target_folder, 'micro_static_aligned_croped', emotion_category, fileame_folder)
                    # folder_path = os.path.join(self.image_root, target_folder, 'micro', emotion_category, fileame_folder)
                    global_flow_path = os.path.join(folder_path, 'flow28.png')
                    patch_flow_path = os.path.join(folder_path, 'flow128.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.global_flow_test.append(global_flow_path)
                    self.patch_flow_test.append(patch_flow_path)
                    self.au_test.append(au_path)
                    self.label_test.append(int(sub_folder))
            self.train_data = self._preload_data(self.global_flow_train, self.patch_flow_train,
                                                 self.au_train) if self.train else []
            self.test_data = self._preload_data(self.global_flow_test, self.patch_flow_test,
                                                self.au_test) if not self.train else []

    def _preload_data(self, global_flow_paths, patch_flow_paths, au_paths):
        data = []
        for global_flow_path, patch_flow_path, au_path in zip(global_flow_paths, patch_flow_paths, au_paths):
            optical_flow = cv2.imread(patch_flow_path)
            points = detect_landmarks_v2(global_flow_path)
            patches = []
            for point in points:
                start_x, end_x, start_y, end_y = get_patches(point)
                patch = optical_flow[start_x:end_x, start_y:end_y]
                patch = cv2.resize(patch, (5, 5), interpolation=cv2.INTER_LINEAR)
                patch = transforms.ToTensor()(patch)
                patches.append(patch)
            patches = torch.cat(patches, dim=0)
            au_list = get_non_empty_numbers_from_first_column(au_path)
            vector_length = len(self.label_mapping)
            au_vector = [0] * vector_length
            for au in au_list:
                if au in self.label_mapping:
                    index = self.label_mapping[au]
                    au_vector[index] = 1
            flow = self.transforms(cv2.imread(global_flow_path))
            data.append((flow, patches, torch.tensor(au_vector)))

        return data

    def __len__(self):
        if self.train:
            return len(self.global_flow_train)
        else:
            return len(self.global_flow_test)

    def __getitem__(self, idx: int):
        if self.train:
            optical_flow, patches, au_vector = self.train_data[idx]
            label = self.label_train[idx]
        else:
            optical_flow, patches, au_vector = self.test_data[idx]
            label = self.label_test[idx]
        return optical_flow, patches, au_vector, torch.tensor(label)


class AugmentedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.transform1 = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 3)),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomRotation(degrees=(-3, 0)),
        ])

        self.transform3 = transforms.Compose([
            transforms.RandomRotation(degrees=(-3, 3)),
        ])
        self.augmented_data = []
        self._augment_data()

    def _augment_data(self):
        for flow, patch, au, label in self.original_dataset:
            if label == 6 or label == 4 or label == 3:
                augmented_flow = self.transform3(flow)
                augmented_patch = self.transform3(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
            if label == 5 or label == 1 or label == 0:
                augmented_flow = self.transform1(flow)
                augmented_patch = self.transform1(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform2(flow)
                augmented_patch = self.transform2(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))



    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            return self.augmented_data[idx - len(self.original_dataset)]


class AugmentedDataset_cas3_3_class(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.transform1 = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 3)),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomRotation(degrees=(-3, 0)),
        ])

        self.transform3 = transforms.Compose([
            transforms.RandomRotation(degrees=(-6, 6)),
        ])

        self.transform4 = transforms.Compose([
            transforms.RandomRotation(degrees=(3, 6)),
        ])

        self.transform5 = transforms.Compose([
            transforms.RandomRotation(degrees=(-6, -3)),
        ])

        self.transform6 = transforms.Compose([
            transforms.RandomRotation(degrees=(6, 9)),
        ])

        self.transform7 = transforms.Compose([
            transforms.RandomRotation(degrees=(-9, -6)),
        ])
        self.augmented_data = []
        self._augment_data()

    def _augment_data(self):
        for flow, patch, au, label in self.original_dataset:
            if label == 2:
                augmented_flow = self.transform3(flow)
                augmented_patch = self.transform3(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
            if label == 1:
                augmented_flow = self.transform1(flow)
                augmented_patch = self.transform1(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform2(flow)
                augmented_patch = self.transform2(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform4(flow)
                augmented_patch = self.transform4(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform5(flow)
                augmented_patch = self.transform5(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform6(flow)
                augmented_patch = self.transform6(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))
                augmented_flow = self.transform7(flow)
                augmented_patch = self.transform7(patch)
                self.augmented_data.append((augmented_flow, augmented_patch, au, label))



    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            return self.augmented_data[idx - len(self.original_dataset)]

