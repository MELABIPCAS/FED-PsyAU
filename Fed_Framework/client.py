import copy
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
from utils.dataset import MEDataset


def confusionMatrix(gt, pred, show=False):
    cm = confusion_matrix(gt, pred)
    if cm.size != 4:
        TN, FP, FN, TP = 0, 0, 0, 0
    else:
        TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN + 1e-22)
    num_samples = len([x for x in gt if x == 1])
    if show:
        print("TP:", TP, "num_samples:", num_samples)
    average_recall = TP / (num_samples + 1e-22)
    return f1_score, average_recall


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


def get_val_loss(args, model, Val, epoch):
    device = args.device
    model.eval()
    total_loss = 0.0
    all_emo_predicts, all_emo_labels = [], []
    au_criterion = nn.BCEWithLogitsLoss().to(device)
    emo_criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for data in Val:
            global_features, au_features, au_labels, emo_labels = data
            global_features, au_features, au_labels, emo_labels = global_features.to(
                device).float(), au_features.to(device).float(), au_labels.to(
                device).float(), emo_labels.to(
                device)

            emo_predicts, au_predicts, auxiliary_au_predicts = model(global_features, au_features, args.ratio)

            au_loss = au_criterion(au_predicts, au_labels) + au_criterion(auxiliary_au_predicts, au_labels)
            emo_loss = emo_criterion(emo_predicts, emo_labels.long())
            total_loss += 0.2 * emo_loss.item() + 0.8 * au_loss.item()
            # total_loss += emo_loss.item()
            all_emo_predicts.extend(torch.max(emo_predicts, 1)[1].tolist())
            all_emo_labels.extend(emo_labels.long().tolist())

        average_val_loss = total_loss / len(Val)  #
        f1_list, recall_list, emo_UF1, emo_UAR = recognition_evaluation(all_emo_labels, all_emo_predicts, show=False)
        cur_acc = sum(x == y for x, y in zip(all_emo_predicts, all_emo_labels)) / len(all_emo_predicts)
        print(f'epoch:{epoch + 1}, test_loss：{average_val_loss}, acc：{cur_acc}, uf1：{emo_UF1}, uar：{emo_UAR}')
        return average_val_loss, cur_acc, emo_UF1, emo_UAR


def train(args, model, server):
    torch.manual_seed(args.seed)
    model.train()
    label_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 12: 8, 14: 9, 15: 10, 17: 11}
    mu = args.mu
    if model.name == 'casme2':
        df = pd.read_excel('data_coding/CASME2-coding.xlsx', engine='openpyxl', dtype={'Subject': str})
        total_dataset = MEDataset(df, label_mapping, 'data/casme2_onset_crop', 'CASME2', args.device, True, [])
        mu = 0.01
    elif model.name == 'samm':
        df = pd.read_excel('data_coding/SAMM-coding.xlsx', engine='openpyxl', dtype={'Subject': str})
        total_dataset = MEDataset(df, label_mapping, 'data/samm_onset_crop', 'SAMM', args.device, True, [])
        mu = 0.001
    elif model.name == 'casme3':
        df = pd.read_excel('data_coding/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx', engine='openpyxl', dtype={'Subject': str})
        total_dataset = MEDataset(df, label_mapping, 'data/casme^3_me_onset_crop', 'CASME3', args.device, True, [])
        mu = 0.01
    total_length = len(total_dataset)
    train_size = int(0.7 * total_length)
    test_size = total_length - train_size
    train_set, test_set = random_split(total_dataset, [train_size, test_size])
    Dtr = DataLoader(train_set, batch_size=args.B, shuffle=True)
    Dte = DataLoader(test_set, batch_size=args.B, shuffle=True)
    device = args.device
    model.len = len(train_set)
    global_model = copy.deepcopy(server)


    lr = args.lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


    def lr_lambda(epoch):
        if epoch < 10:
            return float(epoch) / 10
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f'train_{model.name}')
    print('train_len:', len(train_set))
    print('test_len:', len(test_set))
    au_criterion = nn.BCEWithLogitsLoss().to(device)
    emo_criterion = nn.CrossEntropyLoss().to(device)
    acc, emo_UF1, emo_UAR = 0, 0, 0
    train_loss, val_loss = 0.0, 0.0
    for epoch in tqdm(range(args.E)):
        model.train()
        train_loss = 0.0
        for i, data in enumerate(Dtr, 0):
            global_features, au_features, au_labels, emo_labels = data
            global_features, au_features, au_labels, emo_labels = global_features.to(
                device).float(), au_features.to(device).float(), au_labels.to(device).float(), emo_labels.to(
                device)
            optimizer.zero_grad()
            emo_predicts, au_predicts, auxiliary_au_predicts = model(global_features, au_features, args.ratio)
            au_loss = au_criterion(au_predicts, au_labels) + au_criterion(auxiliary_au_predicts, au_labels)
            emo_loss = emo_criterion(emo_predicts, emo_labels.long())
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)

            total_loss = 0.2 * emo_loss + 0.8 * au_loss + (mu / 2) * proximal_term
            # total_loss = 0.2 * emo_loss + 0.8 * au_loss
            train_loss += au_loss.item() + emo_loss.item()
            total_loss.backward()
            optimizer.step()
        train_loss = train_loss / len(Dtr)
        val_loss, acc, emo_UF1, emo_UAR = get_val_loss(args, model, Dte, epoch)
        warmup_scheduler.step()
    return model, acc, emo_UF1, emo_UAR, train_loss, val_loss





