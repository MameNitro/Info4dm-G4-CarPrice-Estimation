#属性normalized-lossesの欠損値を補完するプログラム(他の欠損値の補完は未実装)

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#データセット読み込み(ダミーコーディング済み)
df = pd.read_csv("Automobile price data _Raw_.csv")

#?をNaN(null)に置換
df = df.replace("?", np.NaN)

#ダミーコーディング
dummy_df = pd.get_dummies(df[["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]])

#dfの連結
df = pd.concat((df, dummy_df), axis=1)

#文字列の含まれる列を削除
drop_col = ["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"]
df = df.drop(drop_col, axis=1)

#推定したい欠損値(normalized-losses)以外の欠損を含むデータ(行)を削除
#"num-of-doors", 
df = df.dropna(subset=["bore", "horsepower", "peak-rpm", "price"])

#推定したい属性(normalized-losses)のデータフレーム作成
df1 = df["normalized-losses"]

#欠損データの個数
# print(df.isnull().sum())

#normalized-losses以外のデータのデータフレーム作成
df_X = df.drop("normalized-losses",axis=1)

#欠損データの番号
index_number = df[df.isnull().any(axis=1)].index

#欠損値をテストデータに、それ以外を学習データにする
df_X_train = df_X.drop(index_number, axis=0)
df_X_test = df_X[df.isnull().any(axis=1)]

#予測するnormalized-lossesの欠損値をテストデータに、それ以外を学習データにする
df_y_train = df1.drop(index_number, axis=0)
df_y_test = df1[df.isnull().any(axis=1)]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(df_X_train, df_y_train)

# Make predictions using the testing set
df_y_pred = regr.predict(df_X_test)

#予測した値(normalized-losses)をデータフレームに入力
for m,n in zip(index_number,df_y_pred):
    df.at[m, "normalized-losses"] = n

#csvに書き出し
df.to_csv("datasets4.csv")







# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(df_y_test, df_y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(df_y_test, df_y_pred))

# # Plot outputs
# plt.scatter(df_X_test, df_y_test,  color='black')
# plt.plot(df_X_test, df_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()