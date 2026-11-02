"""从 DFME 编码表生成每个样本的 au.csv。"""

import argparse
import os

import pandas as pd

FOLDERS = {
    "USTC_MEv4_partA_sub0001_sub0015": range(1, 16),
    "USTC_MEv4_partB_sub0016_sub0110": range(16, 111),
    "USTC_MEv4_partC_sub0111_sub0300": range(111, 301),
    "USTC_MEv4_partD_sub0301_sub0671": range(301, 672),
}


def part_folder(sub_id, fps):
    for folder, sub_range in FOLDERS.items():
        if sub_id in sub_range:
            return f"{folder}_{int(fps)}fps" if "partD" in folder else folder
    return None


def main():
    parser = argparse.ArgumentParser(description="生成 au.csv")
    parser.add_argument("excel", help="DFME_data_coding.xlsx")
    parser.add_argument("data_path", help="dfme_augment 根目录")
    args = parser.parse_args()

    table = pd.read_excel(args.excel, engine="openpyxl",
                          dtype={"Subject": str, "Filename": str, "AU_Split": str})
    for _, row in table.iterrows():
        if str(row["Emotion"]) == "others":
            continue
        subject_id = int(row["Subject"].replace("sub", ""))
        folder = part_folder(subject_id, row["fps"])
        if folder is None:
            continue
        out_dir = os.path.join(args.data_path, folder, "micro_static_aligned_croped",
                               str(row["Emotion"]), row["Filename"])
        os.makedirs(out_dir, exist_ok=True)
        codes = [code.strip() for code in str(row["AU_Split"]).split(",")]
        pd.DataFrame(codes, columns=["au"]).to_csv(os.path.join(out_dir, "au.csv"), index=False)


if __name__ == "__main__":
    main()
