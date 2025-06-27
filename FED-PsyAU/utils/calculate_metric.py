import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score, accuracy_score



def au_evaluate_12(all_au_predicts, all_au_labels, show=False):
    all_au_predicts = all_au_predicts.cpu().detach().numpy()
    all_au_labels = all_au_labels.cpu().detach().numpy()

    au_recall_list = [0] * 12
    au_f1_score_list = [0] * 12
    au_accuracy_list = [0] * 12

    all_au_predicts[all_au_predicts > 0.5] = 1
    all_au_predicts[all_au_predicts <= 0.5] = 0
    for i in range(12):
        au_f1_score_list[i], au_recall_list[i] = confusionMatrix(all_au_labels[:, i], all_au_predicts[:, i], show)
        au_accuracy_list[i] = accuracy_score(all_au_labels[:, i], all_au_predicts[:, i])

    return au_recall_list, au_f1_score_list, au_accuracy_list

def calculate_metrics(all_emo_predicts, all_emo_labels):
    f1_list, recall_list, emo_UF1, emo_UAR = recognition_evaluation(all_emo_labels, all_emo_predicts)
    return f1_list, recall_list, emo_UF1, emo_UAR


def confusionMatrix(gt, pred, show=False):
    cm = confusion_matrix(gt, pred)
    if cm.size != 4:
        TN, FP, FN, TP = 0, 0, 0, 0
    else:
        TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP) / (2*TP + FP + FN + 1e-22)
    num_samples = len([x for x in gt if x == 1])
    if show:
        print("TP:", TP, "num_samples:", num_samples)
    average_recall = TP / (num_samples+1e-22)
    return f1_score, average_recall



def recognition_evaluation_7class(final_gt, final_pred, show=False):
    label_dict = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    f1_list = [0, 0, 0, 0, 0, 0, 0]
    recall_list = [0, 0, 0, 0, 0, 0, 0]

    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, recall_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list[emotion_index] = f1_recog
                recall_list[emotion_index] = recall_recog
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(recall_list)
        return f1_list, recall_list, UF1, UAR
    except:
        return '', ''



def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    f1_list = [0, 0, 0]
    recall_list = [0, 0, 0]

    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, recall_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list[emotion_index] = f1_recog
                recall_list[emotion_index] = recall_recog
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(recall_list)
        return f1_list, recall_list, UF1, UAR
    except:
        return '', ''



