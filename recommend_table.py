import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from common.utility import cos_similarity
import data

"""過去データ取得"""
g_data = np.load('./g_sample_data.npy')
h_data = np.load('./h_sample_data.npy')
#print(g_data.shape)
#print(h_data.shape)


"""new_data取得"""
#new_data import
#仮
a = g_data[4][1]
b = g_data[0][1]
c = g_data[5][2]
new_data = np.concatenate([[a],[b],[c]])


"""各tableのdiwtanceを計算"""
def table_distance(g_data, h_data, new_data):
    """ 
    1) 過去データ：g_data、　+1日後のデータ：h_data、　直前のデータ：new_data
    2) new_dataと、過去データを比較しコサイン類似度を算出
    3) 類似度とnew_dataの要素積をかける（new_dataは-をかけ、その最小値が、「最も類似しかつ+1日後の結果が良い」
    4) tableごとにコサイン類似度の最小値をtable_distanceに格納
    """
    table_distance = {}
    for n  in  range(g_data.shape[0]):
        #コサイン類似度(cos_similarity)
        cos =cos_similarity(g_data[n] , new_data)
        distances = cos  * (- h_data[n].T)
        distance = np.amin(distances)
        table_distance[n] = distance
    best_table = np.argmin(table_distance)
    #return best_table
    return table_distance

"""3次元のnew_dataに対する table_distance関数を使い、best台をレコメンド"""
def predict_table(new_data):
    """
    1) new_dataが2次元配列処理
    2) 全tableから最も良いテーブルをレコメンド
    """
    best_predict = {}
    for table , d in enumerate(new_data):
        best_table = table_distance(g_data, h_data, d)
        predict = sorted(best_table.values())[0]
        best_predict[table] = predict
        
    min_kv = min(best_predict.items(), key=lambda x: x[1])

    return min_kv


print(predict_table(new_data))