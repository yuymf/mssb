# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 20:06
# @Author  : gan
#Q1 PCA+LDA

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from myscore import score
# 读入数据并且分析数据


def show(s):
    print(s)
    print(np.mean(s))

file_location = "作业数据_2021合成.xls"
data = pd.read_excel(file_location)
data.dropna(axis=0, how='any', subset=['肺活量'], inplace=True)
kf = KFold(n_splits = 5,shuffle = True,random_state = None)
se_all=[]
sp_all=[]
acc_all=[]
auc_all=[]
for train_index, test_index in kf.split(data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train =np.array([data["肺活量"].iloc[train_index],data["50米成绩"].iloc[train_index],
              data["身高"].iloc[train_index],data["体重"].iloc[train_index],data["鞋码"].iloc[train_index]])
    x_test =np.array([data["肺活量"].iloc[test_index],data["50米成绩"].iloc[test_index],
             data["身高"].iloc[test_index],data["体重"].iloc[test_index],data["鞋码"].iloc[test_index]])
    y_train, y_test =np.array(data["性别男1女0"].iloc[train_index]),np.array(data["性别男1女0"].iloc[test_index])
    x_train=np.transpose(x_train);x_test=np.transpose(x_test)
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train)
    lda_pred=lda.predict(x_test)
    se,sp,acc,auc =score(lda_pred,y_test)
    se_all.append(se);sp_all.append(sp);acc_all.append(acc);auc_all.append(auc)

show(se_all);show(sp_all);show(acc_all);show(auc_all)