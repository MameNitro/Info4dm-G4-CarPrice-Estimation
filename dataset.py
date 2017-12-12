#モデルの作成を円滑にするために本データを下記の項目に変更
#元データの欠損値(?)をnull(NaN)に変更
#欠損値のあるデータの削除
#文字列が含まれる列の削除

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
# print(df.isnull().sum())

#ダミーコーディング
dummy_df = pd.get_dummies(df[["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]])   
# print(dummy_df)
# dummy_df.to_csv("dummy.csv")

#dfの連結
df = pd.concat((df, dummy_df), axis=1)

#文字列の含まれる列を削除
drop_col = ["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]
df = df.drop(drop_col, axis=1)

# print(df)

#csvに書き出し
df.to_csv("datasets2.csv")
