#datasets.pyでの処理に加えて下記の処理を行なっている
#属性normalized-lossesの欠損値を補完するプログラム(他の欠損値の補完は未実装)

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

#元データ読み込み
df = pd.read_csv("Automobile price data _Raw_.csv").replace("?", np.NaN)

#文字列の含まれる列を削除
drop_col = ["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]
df = df.drop(drop_col, axis=1)

#推定したい欠損値(normalized-losses)以外の欠損を含むデータ(行)を削除
df = df.dropna(subset=["bore", "horsepower", "peak-rpm", "price"])

#推定したい属性(normalized-losses)のデータフレーム作成
df_Y = df["normalized-losses"]

#normalized-losses以外のデータのデータフレーム作成
df_X = df.drop("normalized-losses",axis=1)

#欠損データの番号
index_number = df[df.isnull().any(axis=1)].index

#欠損値をテストデータに、それ以外を学習データにする
df_X_train = df_X.drop(index_number, axis=0)
df_X_test = df_X[df.isnull().any(axis=1)]

#予測するnormalized-lossesの欠損値をテストデータに、それ以外を学習データにする
df_Y_train = df_Y.drop(index_number, axis=0)
df_Y_test = df_Y[df.isnull().any(axis=1)]

#線形回帰モデルの作成
regr = linear_model.LinearRegression()

#学習
regr.fit(df_X_train, df_Y_train)

#予測
df_Y_pred = regr.predict(df_X_test)

#予測した値(normalized-losses)をデータフレームに入力
for m,n in zip(index_number,df_Y_pred):
    df.at[m, "normalized-losses"] = n

#csvに書き出し
df.to_csv("datasets3.csv")
