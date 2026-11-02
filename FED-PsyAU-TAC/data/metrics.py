"""评估指标：UF1、UAR 和 AU 的逐项指标。"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

EMOTION_NAMES = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
AU_NAMES = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]


def binary_scores(gt, pred):
    cm = confusion_matrix(gt, pred)
    if cm.size != 4:
        return 0.0, 0.0
    tn, fp, fn, tp = cm.ravel()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-22)
    recall = tp / (len([x for x in gt if x == 1]) + 1e-22)
    return f1, recall


def recognition_evaluation(gt, pred):
    """返回逐类 F1、逐类召回、UF1、UAR。"""
    f1_list, recall_list = [], []
    for index in range(len(EMOTION_NAMES)):
        f1, recall = binary_scores([1 if x == index else 0 for x in gt],
                                   [1 if x == index else 0 for x in pred])
        f1_list.append(f1)
        recall_list.append(recall)
    return f1_list, recall_list, float(np.mean(f1_list)), float(np.mean(recall_list))


def au_evaluation(au_predicts, au_labels):
    """由 sigmoid 概率算每个 AU 的召回、F1、准确率。"""
    predicts = au_predicts.detach().cpu().numpy()
    labels = au_labels.detach().cpu().numpy()
    predicts = (predicts > 0.5).astype(int)
    recall_list, f1_list, accuracy_list = [], [], []
    for i in range(len(AU_NAMES)):
        f1, recall = binary_scores(labels[:, i], predicts[:, i])
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy_score(labels[:, i], predicts[:, i]))
    return recall_list, f1_list, accuracy_list
