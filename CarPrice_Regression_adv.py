#
# 不要列の捜索，評価用
# １行ずつ見ない場合の誤差値を各々出す
#
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#データセットの読み込み 
Automobile = pd.read_csv('datasets4.csv')
del Automobile['Unnamed: 0']
Automobile_X_origin = Automobile.drop("price",axis=1)

# 最大値，最小値を返すための変数の初期化
MAX_AllAbsoluteError = 0
MIN_AllAbsoluteError = 1145141919810
MAX_NAME = None
MIN_NAME = None

for j in range(len(Automobile_X_origin.columns)):

    #最終誤差値用の結果を格納する配列を用意
    #そして初期化
    AllAbsoluteError = []
    AllSquaredError = []

    #1列削除，及びその1列削除済みデータを読み込み
    print('###Deleted Columns NAME => "{}"'.format(Automobile_X_origin.columns[j]))
    Automobile_X = Automobile_X_origin.drop(Automobile_X_origin.columns[j], axis=1)

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

        # 平均絶対誤差
        AllAbsoluteError.append(mean_absolute_error(Automobile_Y_test, Automobile_Y_pred))
        # 平均二乗誤差
        AllSquaredError.append(np.sqrt(mean_squared_error(Automobile_Y_test, Automobile_Y_pred)))

    # 平均絶対誤差および平均二乗誤差を表示
    print('AbsoluteError => {}'.format(sum(AllAbsoluteError)/len(AllAbsoluteError)))
    print('AquaredError  => {}'.format(sum(AllSquaredError)/len(AllSquaredError)))

    # 最大値を格納
    if MAX_AllAbsoluteError < (sum(AllAbsoluteError)/len(AllAbsoluteError)):
        MAX_AllAbsoluteError = (sum(AllAbsoluteError)/len(AllAbsoluteError))
        MAX_NAME = j

    # 最小値を格納
    if MIN_AllAbsoluteError > (sum(AllAbsoluteError)/len(AllAbsoluteError)):
        MIN_AllAbsoluteError = (sum(AllAbsoluteError)/len(AllAbsoluteError))
        MIN_NAME = j

# 最大値と最小値だったカラムを表示
print('MAX AbsoluteError = {0} {1}'.format(MAX_AllAbsoluteError, Automobile_X_origin.columns[MAX_NAME]))
print('MIN AbsoluteError = {0} {1}'.format(MIN_AllAbsoluteError, Automobile_X_origin.columns[MIN_NAME]))
