import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("ribonn_human.xlsx")
df["mean_te"].hist(bins=100)
plt.savefig('TE_distribution.png')
