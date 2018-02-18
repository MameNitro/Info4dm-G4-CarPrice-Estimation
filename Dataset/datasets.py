#モデルの作成を円滑にするために元データを下記の項目に変更
#元データの欠損値(?)をnull(NaN)に変更
#欠損値のあるデータの削除
#文字列の含まれる属性の削除

import pandas as pd
import numpy as np

#元データ読み込み
df = pd.read_csv('Automobile price data _Raw_.csv').replace("?", np.NaN)

#欠損データの個数
# print(df.isnull().sum())

#欠損値を含む行を削除
df = df.dropna()

#文字列の含まれる列を削除
drop_col = ["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]
df = df.drop(drop_col, axis=1)

#csvに書き出し
df.to_csv("datasets.csv")
