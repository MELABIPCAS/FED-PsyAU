"""在测试集上评估已保存的模型。"""

import pandas as pd
import torch

from config import parse_args
from data.dataset import split_subjects
from models.fed_psyau import FEDPsyAU
from train import build_loader, evaluate, seed_everything


def main():
    config = parse_args("FED-PsyAU evaluation")
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    data_info = pd.read_excel(config.excel_path, engine="openpyxl", dtype={"Subject": str, "Filename": str})

    subjects = list(data_info["Subject"].dropna().unique())
    _, test_subjects = split_subjects(subjects, config.test_ratio, config.seed)
    test_loader = build_loader(data_info, test_subjects, config, triple_frames=False, shuffle=False)

    model = FEDPsyAU(config.num_classes).to(device)
    model.load_state_dict(torch.load(config.checkpoint, map_location=device))

    metrics = evaluate(model, test_loader, device)
    print(f"UF1 {metrics['UF1']:.4f}  UAR {metrics['UAR']:.4f}")
    for name, f1, recall in zip(["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"],
                                metrics["f1"], metrics["recall"]):
        print(f"  {name:<10} F1 {f1:.4f}  Recall {recall:.4f}")
    print("AU F1:", [round(value, 4) for value in metrics["au_f1"]])


if __name__ == "__main__":
    main()
