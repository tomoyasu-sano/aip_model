import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data

"""
data整形手順
1) excel import
2) 2次元をdata(台)を、時系列データを変換(+1日後を予測)
　＊この時系列データ変換は、当初はLSTMに学習させるための入力データの準備であった
3) 奥行きを入れた3次元データに変換し時系列dataの完成
"""



"""時系列データの作成関数"""
def make_dataset(low_data, n_prev=100):

    data, target = [], []
    # 21データずつ分割 == 下記のcと同じ値とする
    maxlen = 21

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target



"""import data 仮で3次元データの作成"""
#test_data = pd.DataFrame(np.random.randint(1,100, (25, 50)))
#test_data.head()

"""エクセルデータより"""
test_data = pd.read_excel('./data/sample_data.xlsx')
table = test_data["tables"]
test_data = test_data.drop("tables", axis=1)
#print(test_data.shape)
#test_data.head()

"""時系列データの整形時の処理準備"""
# cはmaxlenと同じ値
c = 21
r = test_data.shape[1] - c

#g_data = pd.DataFrame()
#h_data = pd.DataFrame()
h_data = np.empty([1, r,1])
g_data = np.empty([1, r, c])
for index , row in test_data.iterrows():
    #male_dataset関数
    g, h = make_dataset(row)
    
    g_add = g.reshape((1, r,c))
    g_data = np.r_[g_data, g_add]
    
    h_add = h.reshape((1, r,1))
    h_data = np.r_[h_data, h_add]

g  = np.delete(g_data , 0, 0)
h  = np.delete(h_data , 0, 0)


np.save('./g_sample_data', g)
np.save('./h_sample_data', h)