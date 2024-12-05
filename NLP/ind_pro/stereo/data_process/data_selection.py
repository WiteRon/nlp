# coding:utf-8
import pandas as pd

# 读取原始CSV文件
original_csv_path = '/home/zshiap/NLP/ind_pro/stereo/crows-pairs/data/crows_pairs_anonymized.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(original_csv_path)

# 筛选出bias_type列为gender的行
gender_bias_df = df[df['bias_type'] == 'gender']

# 将筛选后的DataFrame保存为新的CSV文件
new_csv_path = '/home/zshiap/NLP/ind_pro/stereo/data_process/selected.csv'  # 替换为你想要保存的新CSV文件路径
gender_bias_df.to_csv(new_csv_path, index=False)

print(f"csv saved as:{'/home/zshiap/NLP/ind_pro/stereo/data_process/selected.csv'}")


