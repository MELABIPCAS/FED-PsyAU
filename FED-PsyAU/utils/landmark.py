
import dlib
import matplotlib.pyplot as plt
import os
import csv
import cv2
import json
import math
import numpy as np

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../weight/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
EYEBROW_LEFT_INDEX = (17, 22)
EYEBROW_RIGHT_INDEX = (22, 27)
EYE_LEFT_INDEX = (36, 42)
EYE_RIGHT_INDEX = (42, 48)
MOUTH_INDEX = (48, 68)
CHIN_INDEX = (6, 11)
NOSE_INDEX = (27, 36)

def crop_picture(img_rd,size):
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
    left=landmarks[39]
    right=landmarks[42]
    # print(left)
    # print(right)
    thete=math.atan(float(right[0,1]-left[0,1])/(right[0,0]-left[0,0]))


    gezi=int((right[0,0]-left[0,0])/2)
    center=[int((right[0,0]+left[0,0])/2),int((right[0,1]+left[0,1])/2)]
    # print(8*gezi)


    cv2.rectangle(img_rd, (center[0] - int(4.5 * gezi), center[1] - int(3.5 * gezi)), (center[0] + int(4.5 * gezi), center[1] + int(5.5 * gezi)),
                  (0, 0, 255), 2)

    a=(center[1] - int(3 * gezi))
    b=center[1] +int(5 * gezi)
    c=(center[0] - int(4 * gezi))
    d=center[0] +int(4 * gezi)

    a=max((center[1] - int(3 * gezi)),0)
    # b=min(center[1] +int(5.5 * gezi),399)
    c=max(center[0] - int(4 * gezi),0)
    # d=min(center[0] +int(4.5 * gezi),399)
    img_crop = img_rd[a:b, c:d]
    # print(img_crop.shape)
    # cv2.imshow("image", img_crop)
    # cv2.waitKey(0)
    img_crop_samesize = cv2.resize(img_crop, (size, size))
    return landmarks, img_crop_samesize, a, b, c, d

def read_detect_faces(csv_path):

    detect_faces = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            left, top, right, bottom = map(int, row)
            detect_faces.append(dlib.rectangle(left, top, right, bottom))
    return detect_faces

def read_landmarks(landmarks_path):
    if not os.path.exists(landmarks_path):
        print(f"Error: File '{landmarks_path}' not found.")
        return None
    with open(landmarks_path, 'r') as f:
        landmarks = json.load(f)
    return landmarks

def detect_landmarks(img_path):
    landmarks_path = os.path.join(os.path.dirname(img_path), 'landmarks.json')
    landmarks = read_landmarks(landmarks_path)
    return landmarks

def detect_landmarks_v2(img_path):
    landmarks_path = os.path.join(os.path.dirname(img_path), 'landmarks_v2.json')
    landmarks = read_landmarks(landmarks_path)
    return landmarks

def plot_landmarks(img, landmarks):
    plt.imshow(img)
    x, y = zip(*landmarks)
    plt.scatter(x, y, s=20, marker='.', c='r')
    plt.show()
