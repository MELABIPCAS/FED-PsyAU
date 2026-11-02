"""训练入口。"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import LABEL_MAPPING, parse_args
from data.dataset import AugmentedDataset, MEDataset, split_subjects
from data.metrics import au_evaluation, recognition_evaluation
from models.fed_psyau import FEDPsyAU, mer_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(data_info, subjects, config, triple_frames, shuffle):
    dataset = MEDataset(data_info, LABEL_MAPPING, config.data_path, subjects, triple_frames)
    if triple_frames:
        dataset = AugmentedDataset(dataset)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def to_device(batch, device):
    global_flow, patches, points, au_labels, emo_labels = batch
    return (global_flow.to(device).float(), patches.to(device).float(), points.to(device).float(),
            au_labels.to(device).float(), emo_labels.to(device).long())


def evaluate(model, loader, device):
    model.eval()
    emo_predicts, emo_labels, au_predicts, au_labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            global_flow, patches, points, au_label, emo_label = to_device(batch, device)
            main_logits, _, _, au_logits_gat = model(global_flow, patches, points)
            emo_predicts.extend(main_logits.argmax(dim=1).tolist())
            emo_labels.extend(emo_label.tolist())
            au_predicts.append(torch.sigmoid(au_logits_gat))
            au_labels.append(au_label)

    f1_list, recall_list, uf1, uar = recognition_evaluation(emo_labels, emo_predicts)
    au_recall, au_f1, au_acc = au_evaluation(torch.cat(au_predicts), torch.cat(au_labels))
    return {"UF1": uf1, "UAR": uar, "f1": f1_list, "recall": recall_list,
            "au_f1": au_f1, "au_recall": au_recall, "au_acc": au_acc}


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        global_flow, patches, points, au_labels, emo_labels = to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(global_flow, patches, points)
        loss, _ = mer_loss(outputs, emo_labels, au_labels, config.alpha1, config.alpha2, config.alpha3)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def run_once(data_info, config, device, seed):
    seed_everything(seed)
    subjects = list(data_info["Subject"].dropna().unique())
    train_subjects, test_subjects = split_subjects(subjects, config.test_ratio, seed)
    train_loader = build_loader(data_info, train_subjects, config, triple_frames=True, shuffle=True)
    test_loader = build_loader(data_info, test_subjects, config, triple_frames=False, shuffle=False)

    model = FEDPsyAU(config.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_uf1, best_uar = 0.0, 0.0
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, config, device)
        print(f"run {seed} epoch {epoch + 1}/{config.epochs} loss {loss:.4f}")
        if (epoch + 1) % config.eval_interval == 0:
            metrics = evaluate(model, test_loader, device)
            print(f"run {seed} epoch {epoch + 1} UF1 {metrics['UF1']:.4f} UAR {metrics['UAR']:.4f}")
            if metrics["UF1"] > best_uf1:
                best_uf1, best_uar = metrics["UF1"], metrics["UAR"]
                torch.save(model.state_dict(), config.checkpoint)
    return best_uf1, best_uar


def main():
    config = parse_args("FED-PsyAU training")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if os.path.dirname(config.checkpoint):
        os.makedirs(os.path.dirname(config.checkpoint), exist_ok=True)
    data_info = pd.read_excel(config.excel_path, engine="openpyxl", dtype={"Subject": str, "Filename": str})

    uf1_list, uar_list = [], []
    for run in range(config.runs):
        uf1, uar = run_once(data_info, config, device, config.seed + run)
        uf1_list.append(uf1)
        uar_list.append(uar)

    print(f"UF1 {np.mean(uf1_list):.4f} ± {np.std(uf1_list):.4f}")
    print(f"UAR {np.mean(uar_list):.4f} ± {np.std(uar_list):.4f}")


if __name__ == "__main__":
    main()
