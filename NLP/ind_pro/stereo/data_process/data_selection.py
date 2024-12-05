# coding:utf-8
import pandas as pd

# ��ȡԭʼCSV�ļ�
original_csv_path = '/home/zshiap/NLP/ind_pro/stereo/crows-pairs/data/crows_pairs_anonymized.csv'  # �滻Ϊ���CSV�ļ�·��
df = pd.read_csv(original_csv_path)

# ɸѡ��bias_type��Ϊgender����
gender_bias_df = df[df['bias_type'] == 'gender']

# ��ɸѡ���DataFrame����Ϊ�µ�CSV�ļ�
new_csv_path = '/home/zshiap/NLP/ind_pro/stereo/data_process/selected.csv'  # �滻Ϊ����Ҫ�������CSV�ļ�·��
gender_bias_df.to_csv(new_csv_path, index=False)

print(f"csv saved as:{'/home/zshiap/NLP/ind_pro/stereo/data_process/selected.csv'}")


