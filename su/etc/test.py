import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ws = 'packages/src/'
# csvの読み込み
sample_data_set = pd.read_csv(ws+'test.csv', sep=';')

# グラフ化してみる
plt.hist(sample_data_set['age'], bins=[15,16,17,18,19,20,21,22])
# 縦軸と横軸に名前を
plt.xlabel('age')
plt.ylabel('count')
plt.savefig(ws+'tmp.png')