#
# 不要列の捜索，評価用
# １行ずつ見ない場合の誤差値を各々出す
#
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#データセットの読み込み 
Automobile = pd.read_csv('datasets4.csv')
Automobile_X_origin = Automobile.drop("price",axis=1)

for j in range(len(Automobile_X_origin.columns)):

    #最終誤差値用の結果を格納する配列を用意
    #そして初期化
    AllAbsoluteError = []
    AllSquaredError = []

    #1列削除，及びその1列削除済みデータを読み込み
    print('###Deleted Columns NAME => "{}"'.format(Automobile_X_origin.columns[j]))
    Automobile_X = Automobile_X_origin.drop(Automobile_X_origin.columns[j], axis=1)

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

        # The coefficients 係数？
        # print('Coefficients: \n', regr.coef_)

        # Explained variance score: 1 is perfect prediction
        # 分散スコア→１なら完全予測
        # print('Variance score: %.2f' % r2_score(Automobile_y_test, Automobile_y_pred))

        # The mean absolute error
        # 平均絶対誤差
        # print('Mean Absolute error: %.2f' 
        #      % mean_absolute_error(Automobile_y_test, Automobile_y_pred))
        AllAbsoluteError.append(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred))

        AllSquaredError.append(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred)))
        # The mean squared error 
        # 平均二乗誤差
        # print("Mean squared error: %.2f"
        #      % mean_squared_error(Automobile_y_test, Automobile_y_pred))
    # for n in range(len(AllAbsoluteError)) :
    #     print(AllAbsoluteError[n])

    print('AbsoluteError => {}'.format(sum(AllAbsoluteError)/len(AllAbsoluteError)))
    print('AquaredError  => {}'.format(sum(AllSquaredError)/len(AllSquaredError)))
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
