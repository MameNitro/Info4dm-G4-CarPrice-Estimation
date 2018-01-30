#
# 線形回帰モデル．
# 評価方法はクロスバリデーション（確認交差法）．
# なので，平均絶対誤差と平均二乗誤差は全て同じ値となる．
#
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

#データセットの読み込み 
Automobile = pd.read_csv('datasets4.csv')
del Automobile['Unnamed: 0']
Automobile_X = Automobile.drop("price",axis=1)

#最終誤差値用の結果を格納する配列を用意
AllAbsoluteError = []
AllSquaredError = []

# クロスバリデーション（確認交差法）
for i in range(len(Automobile_X)):
    
    #テスト用データを１行だけ指定
    Automobile_X_test = Automobile_X[i:i+1]
    Automobile_Y_test = Automobile.price[i:i+1]

    #テストデータとして指定した１行を除いた他の行を学習データとして指定
    Automobile_X_train = Automobile_X.drop(i)
    Automobile_Y_train = Automobile.price.drop(i)

    # 線形モデルを作成
    regr = linear_model.LinearRegression()

    # †学習†
    regr.fit(Automobile_X_train, Automobile_Y_train)

    # テストデータで予測を作成
    Automobile_Y_pred = regr.predict(Automobile_X_test)

    # 平均絶対誤差を計算して各行の結果を配列へその都度記録
    AllAbsoluteError.append(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred))
    # 平均二乗誤差でも同じことを
    AllSquaredError.append(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred)))

# 各々の平均誤差の平均を取ってその値を出力
print('AbsoluteError => {}'.format(sum(AllAbsoluteError)/len(AllAbsoluteError)))
print('AquaredError  => {}'.format(sum(AllSquaredError)/len(AllSquaredError)))
