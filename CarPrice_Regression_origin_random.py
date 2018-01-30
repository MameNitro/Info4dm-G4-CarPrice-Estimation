#
# ランダムに１０行を選択．それをテストデータとしそれらを用いて評価を行う
# 
from numpy.random import *
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#シード値を指定
seed(200)

#データセットの読み込み 
Automobile = pd.read_csv('datasets4.csv')
print(Automobile.shape)
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
regr = RandomForestClassifier()

# †学習†
regr.fit(Automobile_X_train, Automobile_Y_train)

# テストデータで予測を作成
Automobile_Y_pred = regr.predict(Automobile_X_test)

#平均絶対誤差と平均二乗誤差を出すよ
print("AbsoluteError => {}".format(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred)))
print("SquaredError  => {}".format(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred))))

### クロスバリデーションのときの合計誤差を出す行の名残
#AllAbsoluteError.append(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred))
#AllSquaredError.append(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred)))
#print('AbsoluteError => {}'.format(sum(AllAbsoluteError)/len(AllAbsoluteError)))
#print('AquaredError  => {}'.format(sum(AllSquaredError)/len(AllSquaredError)))
