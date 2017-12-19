# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Automobile dataset
Automobile = pd.read_csv('datasets2.csv')
Automobile_X = Automobile.drop("price",axis=1)
### １列目だけを取得 ###
### Automobile.iloc[:,1] ###
### ０列目のデータを削除
### print(Automobile_X.drop(Automobile_X.columns[[0]],axis=1))

for j in range(len(Automobile_X.columns)):

    #---# １列を見ずにやるのを全部出す #---#
    print(Automobile_X.columns[j])
    Automobile_X_d = Automobile_X.drop(Automobile_X.columns[[j]],axis=1)

    # 交差確認法用の合計データの配列
    AllAbsoluteError = []

    for i in range(len(Automobile_X_d)):
        
        Automobile_y_test = Automobile.price.iloc[i]
        Automobile_X_test = Automobile_X_d.iloc[i]

        # 学習に使うデータ行とテストに使うデータ行を指定
        Automobile_y_train = Automobile.price.drop(i)
        Automobile_X_train = Automobile_X_d.drop(i)

        # 学習に使うターゲット行とテストに使うターゲット行を指定

        # 線形モデルを作成
        regr = linear_model.LinearRegression()

        # †学習†
        regr.fit(Automobile_X_train, Automobile_y_train)

        # テストデータで予測を作成
        Automobile_y_pred = regr.predict(Automobile_X_test)

        # The coefficients 係数？
        # print('Coefficients: \n', regr.coef_)

        # Explained variance score: 1 is perfect prediction
        # 分散スコア→１なら完全予測
        # print('Variance score: %.2f' % r2_score(Automobile_y_test, Automobile_y_pred))

        # The mean absolute error
        # 平均絶対誤差
        # print('Mean Absolute error: %.2f' 
        #      % mean_absolute_error(Automobile_y_test, Automobile_y_pred))
        AllAbsoluteError.append(mean_absolute_error(Automobile_y_test, Automobile_y_pred))

        # The mean squared error 
        # 平均二乗誤差
        # print("Mean squared error: %.2f"
        #      % mean_squared_error(Automobile_y_test, Automobile_y_pred))
    # for n in range(len(AllAbsoluteError)) :
    #     print(AllAbsoluteError[n])

    print(sum(AllAbsoluteError)/len(AllAbsoluteError))

# The mean squared logarithmic error
# 平均二乗対数誤差
# print("Mean squared logarithmic error: %.2f" 
#       % mean_squared_log_error(Automobile_y_test, Automobile_y_pred))

# print(Automobile_y_pred)
# print(Automobile_y_test)
# print(Automobile_y_test - Automobile_y_pred)

# Plot outputs
# グラフに描画するゾーン．削除可
# plt.scatter(Automobile_X_test, Automobile_y_test,  color='black')
# plt.plot(Automobile_X_test, Automobile_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
