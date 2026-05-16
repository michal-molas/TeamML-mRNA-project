import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_excel("ribonn_human.xlsx")
# df["mean_te"].hist(bins=100)
# plt.savefig('TE_distribution.png')

# print(df.columns)
# print(df)

df = pd.read_csv("dataset.csv")
print(df.columns)

for col in ['utr3', 'cds', 'utr5']:
    df[col].map(lambda x : len(x)).hist(bins=100)
    plt.savefig(f'{col.upper()}_len_distribution.png')
    plt.close()