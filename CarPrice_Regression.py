# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Autemobile dataset
Autemobile = pd.read_csv('datasets.csv')

Autemobile_X = Autemobile.drop("price",axis=1)

# 学習に使うデータ行とテストに使うデータ行を指定
Autemobile_X_train = Autemobile_X[:-20]
Autemobile_X_test = Autemobile_X[-20:]

# 学習に使うターゲット行とテストに使うターゲット行を指定
Autemobile_y_train = Autemobile.price[:-20]
Autemobile_y_test = Autemobile.price[-20:]

# 線形モデルを作成
regr = linear_model.LinearRegression()

# †学習†
regr.fit(Autemobile_X_train, Autemobile_y_train)

# テストデータで予測を作成
Autemobile_y_pred = regr.predict(Autemobile_X_test)

# The coefficients 係数？
print('Coefficients: \n', regr.coef_)

# Explained variance score: 1 is perfect prediction
# 分散スコア→１なら完全予測
print('Variance score: %.2f' % r2_score(Autemobile_y_test, Autemobile_y_pred))

# The mean absolute error
# 平均絶対誤差
print('Mean Absolute error: %.2f' 
      % mean_absolute_error(Autemobile_y_test, Autemobile_y_pred))

# The mean squared error 
# 平均二乗誤差
print("Mean squared error: %.2f"
      % mean_squared_error(Autemobile_y_test, Autemobile_y_pred))

# The mean squared logarithmic error
# 平均二乗対数誤差
# print("Mean squared logarithmic error: %.2f" 
#       % mean_squared_log_error(Autemobile_y_test, Autemobile_y_pred))

print(Autemobile_y_pred)
print(Autemobile_y_test)
print(Autemobile_y_test - Autemobile_y_pred)

# Plot outputs
# グラフに描画するゾーン．削除可
# plt.scatter(Autemobile_X_test, Autemobile_y_test,  color='black')
# plt.plot(Autemobile_X_test, Autemobile_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
