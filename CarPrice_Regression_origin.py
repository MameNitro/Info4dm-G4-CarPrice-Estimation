#ランダムに１０行を選択．それをテストデータとしそれらを用いて評価を行う
# import matplotlib.pyplot as plt
from numpy.random import *
import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#シード値を指定
seed(100)

#データセットの読み込み 
Automobile = pd.read_csv('datasets4.csv')
del Automobile['Unnamed: 0']
Automobile_X = Automobile.drop("price",axis=1)

col_len = len(Automobile.index) #行数を取得．乱数生成のときの上限に使います
seed_num = 10 #データ行の個数を指定
num_len = [] #生成した乱数を格納する配列．この番号がテストデータの行番号になるよ

#単純な乱数だと重複を許してしまうので，被りがないように仮ソース
while len(num_len) != seed_num:
    num_len = randint(0,col_len,seed_num)
    num_len = set(num_len)

num_len = list(num_len) #重複を回避する過程でset型になるのでlist型に戻すよ
print("TEST Data low -> {}".format(num_len))

#テスト用データを１０行だけ指定
Automobile_X_test = Automobile_X.iloc[num_len,:]
Automobile_Y_test = Automobile.price.iloc[num_len]

#テストデータとして指定した１０行を除いた他の行を学習データとして指定
Automobile_X_train = Automobile_X.drop(num_len)
Automobile_Y_train = Automobile.price.drop(num_len)

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
    # The mean squared error 
    # 平均二乗誤差
    # print("Mean squared error: %.2f"
    #      % mean_squared_error(Automobile_y_test, Automobile_y_pred))
# for n in range(len(AllAbsoluteError)) :
#     print(AllAbsoluteError[n])

#平均絶対誤差と平均二乗誤差を出すよ
print("AbsoluteError => {}".format(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred)))
print("SquaredError  => {}".format(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred))))

### クロスバリデーションのときの合計誤差を出す行の名残
#AllAbsoluteError.append(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred))
#AllSquaredError.append(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred)))
#print('AbsoluteError => {}'.format(sum(AllAbsoluteError)/len(AllAbsoluteError)))
#print('AquaredError  => {}'.format(sum(AllSquaredError)/len(AllSquaredError)))


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
