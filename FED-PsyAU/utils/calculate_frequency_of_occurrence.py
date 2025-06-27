import pandas as pd
from collections import Counter

excel_file = '../data_coding/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'
df = pd.read_excel(excel_file)

data_column = df.iloc[:, 8]

number_counter = Counter()

for item in data_column:
    numbers = map(int, str(item).split(','))
    number_counter.update(numbers)

for number, count in number_counter.most_common():
    print(f"Au{number} 出现了 {count} 次")