#欠損値のあるデータは削除し、書き出し

import pandas as pd
import numpy as np

#テストデータ
# df = pd.DataFrame(  {'age' : ["33", "25", "52"], 
#                     'height' : ["175", "170", "?"],
#                     'weight' : ["70", "?", "60"], 
#                     'job' : ['employee', 'neet', 'employee']}).replace("?", np.NaN)

#本データ
df = pd.read_csv('Automobile price data _Raw_.csv').replace("?", np.NaN)

#?をNaN(null)に置換
df = df.replace("?", np.NaN)

#欠損値を含む行を削除
df = df.dropna()

#欠損データの個数
print(df.isnull().sum())

#csvに書き出し
df.to_csv("datasets.csv")
