# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 08:02:51 2021

@author: Kevin
"""

import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import itertools
import os
import matplotlib
import joblib
#設定matplotlib中文字型
matplotlib.rc("font",family='Microsoft JhengHei')

#繪製confusion_matrix圖
def plot_confusion_matrix(cm, target_names,model_file,cmap=None):
    accuracy = np.trace(cm) / float(np.sum(cm)) #計算準確率
    misclass = 1 - accuracy #計算錯誤率
    if cmap is None:
        cmap = plt.get_cmap('Blues') #顏色設置成藍色
    plt.figure(figsize=(9, 8)) #設置視窗尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #顯示圖片
    plt.title('{}\nConfusion matrix'.format(model_file)) #顯示標題
    plt.colorbar() #繪製顏色條

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names) #x座標
        plt.yticks(tick_marks, target_names, rotation=90) #y座標標籤旋轉90度

    pm = cm.astype('float32') / cm.sum(axis=1)
    pm = np.round(pm,2) #對數字保留兩位元小數

    thresh = cm.max() / 1.5 
    #if normalize else cm.max() / 2
    #將cm.shape[0]、cm.shape[1]中的元素組成元組，遍歷元組中每一個數字
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 

        plt.text(j, i, "{:,} ({:0.2f})".format(cm[i, j],pm[i, j]),
                    horizontalalignment="center",  #數字在方框中間
                    color="white" if cm[i, j] > thresh else "black") #設置字體顏色

    plt.tight_layout() #自動調整子圖參數,使之填充整個圖像區域
    plt.subplots_adjust(left = 0.08, bottom = 0.09)
    plt.ylabel('True label') #y方向上的標籤
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass)) #x方向上的標籤
    #儲存圖片
    if not os.path.exists("confusion_matrix"):
        os.makedirs("confusion_matrix")
    plt.savefig('./confusion_matrix/{}.png'.format(model_file))
    plt.show() #顯示圖片
#資料標準化並將pm2.5欄位分為兩類
def get_dataset(df):
    X = df.iloc[:,1:].to_numpy()
    X = np.nan_to_num(X)
    X = preprocessing.scale(X)
    Z = df.iloc[:,0].to_numpy()
    y = Z.copy()
    #pm2.5 值小於71:適合外出、大於等於71:不適合外出
    for i in range(len(y)):
        if Z[i] < 71:
            y[i] = 0
        else:
            y[i] = 1
    return X, y

#讀取整理過的資料
df = pd.read_csv('./PM2.5/PRSA_data_2010.1.1-2014.12.31.csv') 
#將欄位:pm2.5是空格的刪除
df0 = df.dropna(subset=["pm2.5"]).reset_index()

#將風向字串轉為數字編碼
df_cbwd = df0["cbwd"].drop_duplicates().reset_index()
cbwd = df_cbwd["cbwd"].to_dict()
cbwd_map = dict(zip(cbwd.values(),cbwd.keys()))
df0["cbwd"] = df0["cbwd"].map(cbwd_map)

#計算皮爾森系數
df2 = df0.iloc[:, 3:]
df3 = df2.corr()
#取系數絕對值大於0.06來做為特徵值欄位
df4 = df3['pm2.5'][df3['pm2.5'].abs() >= 0.00].sort_values(ascending=False)
corr_items =  df4.index[:]
print(corr_items)

#存檔
#df3.to_excel('corr.xlsx')
#df4.to_excel('corr_01.xlsx')

#資料集保留2014年做為預測用，其餘做為訓練資料
df_X = df0[df0["year"] != 2014]
df_2014 = df0[df0["year"] == 2014]
X, y = get_dataset(df_X[corr_items])
X_2014, y_2014 = get_dataset(df_2014[corr_items])
print(X_2014)
#建立訓練資料集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, stratify=y)

#建立演算法
SVC = svm.SVC()
knn = KNeighborsClassifier(n_neighbors=5)
xgboostModel = XGBClassifier(n_estimators=1000, learning_rate= 0.3, eval_metric='mlogloss', verbosity=1)
models = {'SVC':SVC, 'knn':knn, 'xgboostModel':xgboostModel}
for key, val in models.items():
    #選擇模型
    model = val
    #訓練模型
    model.fit(X_train,y_train)
    #模型分數
    score = model.score(X_test,y_test)
    
    model_file = 'PM2.5_{}'.format(key) 
    if not os.path.exists("model"):
        os.makedirs("model")
    #儲存訓練完結束的模型
    joblib.dump(model, './model/{}.pkl'.format(model_file))
    #讀取訓練好的模型
    loaded_model = joblib.load('./model/{}.pkl'.format(model_file))
    #預測
    y_pred = loaded_model.predict(X_2014) 
    #正確率
    accuracy = metrics.accuracy_score(y_2014, y_pred)
    cm = metrics.confusion_matrix(y_2014, y_pred)
    print('{}模型分數: {}%'.format(key, round(score*100, 2)))
    print('{}模型準確率: {}%'.format(key, round(accuracy*100, 2)))
    target_names = ['適合外出', '不適合外出']
    plot_confusion_matrix(cm, target_names,model_file,cmap=None)    
  
