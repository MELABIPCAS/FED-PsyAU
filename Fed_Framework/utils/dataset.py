import json
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def read_landmarks(landmarks_path):
    if not os.path.exists(landmarks_path):
        print(f"Error: File '{landmarks_path}' not found.")
        return None
    with open(landmarks_path, 'r') as f:
        landmarks = json.load(f)
    return landmarks

def detect_landmarks_v2(img_path):

    landmarks_path = os.path.join(os.path.dirname(img_path), 'landmarks_v2.json')
    landmarks = read_landmarks(landmarks_path)
    return landmarks
def get_sub_folder(emotion):
    if emotion.lower() == 'happiness' or emotion.lower() == 'happy':
        return '1'
    elif emotion.lower() == 'surprise':
        return '2'
    elif emotion.lower() == 'others' or emotion.lower() == 'other':
        return None
    else:
        return '0'

def get_patches(point: tuple):
    start_x = point[0] - 2
    end_x = point[0] + 3

    start_y = point[1] - 2
    end_y = point[1] + 3

    if start_x == end_x:
        if start_x >= 64 :
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

def get_non_empty_numbers_from_first_column(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        first_column = df.iloc[:, 0]
        non_empty_numbers = [value for value in first_column if pd.notnull(value) and isinstance(value, (int, float))]
        return non_empty_numbers
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

class MyDataset(Dataset):

    def __init__(self, data_info: pd.DataFrame, image_root: str, category: str, device: torch.device, train: bool,
                 test_subject: list):
        self.image_root = image_root
        self.data_info = data_info
        self.test_subject = test_subject
        self.category = category
        self.train = train
        self.device = device
        # self.transforms = transforms.ToTensor()
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.flow_train = []
        self.au_train = []
        self.label_train = []
        self.flow_test = []
        self.au_test = []
        self.label_test = []

        if self.category == 'CASME2':
            for index, row in self.data_info.iterrows():
                subject_folder = 'sub' + row[0]
                fileame_folder = row[1]
                emotion_category = row[8]

                sub_folder = get_sub_folder(emotion_category)
                if sub_folder is not None and subject_folder not in self.test_subject:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    self.flow_train.append(flow_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and subject_folder in self.test_subject:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    self.flow_test.append(flow_path)
                    self.label_test.append(int(sub_folder))
        elif self.category == 'SAMM':
            for index, row in self.data_info.iterrows():
                first_col = row[0]
                second_col = row[1]
                emotion_category = row[9]
                subject_folder = os.path.join(self.image_root, first_col)
                sample_folder = os.path.join(subject_folder, second_col)
                sub_folder = get_sub_folder(emotion_category)
                if sub_folder is not None and first_col not in self.test_subject:
                    flow_path = os.path.join(sample_folder, 'flow28.png')
                    self.flow_train.append(flow_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and first_col in self.test_subject:
                    flow_path = os.path.join(sample_folder, 'flow28.png')
                    self.flow_test.append(flow_path)
                    self.label_test.append(int(sub_folder))

        self.train_data = self._preload_data(self.flow_train) if self.train else []
        self.test_data = self._preload_data(self.flow_test) if not self.train else []

    def _preload_data(self, flow_paths):
        data = []
        for flow_path in flow_paths:
            optical_flow = cv2.imread(flow_path)
            data.append((self.transforms(optical_flow)))
        return data

    def __len__(self):
        if self.train:
            return len(self.flow_train)
        else:
            return len(self.flow_test)

    def __getitem__(self, idx: int):
        if self.train:
            optical_flow = self.train_data[idx]
            label = self.label_train[idx]
        else:
            optical_flow = self.test_data[idx]
            label = self.label_test[idx]
        return optical_flow, torch.tensor(label)



class MEDataset(Dataset):

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, image_root: str, category: str, device: torch.device, train: bool, test_subject_list: list):
        self.image_root = image_root
        self.data_info = data_info
        self.test_subject_list = test_subject_list
        self.label_mapping = label_mapping
        self.category = category
        self.train = train
        self.device = device
        self.transforms = transforms.ToTensor()
        self.flow_train = []
        self.au_train = []
        self.label_train = []
        self.flow_test = []
        self.au_test = []
        self.label_test = []

        if self.category == 'CASME2':
            for index, row in self.data_info.iterrows():
                subject_folder = 'sub' + row[0]
                fileame_folder = row[1]
                emotion_category = row[8]

                sub_folder = get_sub_folder(emotion_category)

                if sub_folder is not None and subject_folder not in self.test_subject_list:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_train.append(flow_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and subject_folder in self.test_subject_list:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_test.append(flow_path)
                    self.au_test.append(au_path)
                    self.label_test.append(int(sub_folder))
        elif self.category == 'SAMM':
            for index, row in self.data_info.iterrows():
                subject_folder = row[0]
                fileame_folder = row[1]
                emotion_category = row[9]

                sub_folder = get_sub_folder(emotion_category)
                if sub_folder is not None and subject_folder not in self.test_subject_list:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_train.append(flow_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                elif sub_folder is not None and subject_folder in self.test_subject_list:
                    folder_path = os.path.join(self.image_root, subject_folder, fileame_folder)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_test.append(flow_path)
                    self.au_test.append(au_path)
                    self.label_test.append(int(sub_folder))
        elif self.category == 'CASME3':
            for index, row in self.data_info.iterrows():
                subject_folder = row[0]
                emotion_category = row[7]
                element1 = row[0]
                element2 = row[1]
                element3 = row[2]
                sub_folder = get_sub_folder(emotion_category)
                if subject_folder not in self.test_subject_list and sub_folder is not None:
                    folder_name = f"{element1}_{element2}_{element3}"
                    folder_path = os.path.join(self.image_root, folder_name)
                    flow_path = os.path.join(folder_path, 'flow28.png')
                    au_path = os.path.join(folder_path, 'au.csv')
                    self.flow_train.append(flow_path)
                    self.au_train.append(au_path)
                    self.label_train.append(int(sub_folder))
                elif subject_folder in self.test_subject_list and sub_folder is not None:
                    folder_name = f"{element1}_{element2}_{element3}"
                    folder_path = os.path.join(self.image_root, folder_name)
                    flow_path = os.path.join(folder_path, 'flow28.png')
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
            flow = self.transforms(global_optical_flow)
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