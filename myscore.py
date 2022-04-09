# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 0:51
# @Author  : gan
#准确性(AC)、敏感性(SE)、特异性(SP) AUC
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def score(y_pred,y_true,label=1):
    confusion_matrix=metrics.confusion_matrix(y_true,y_pred)
    FP = confusion_matrix .sum(axis=0) - np.diag(confusion_matrix )
    FN = confusion_matrix .sum(axis=1) - np.diag(confusion_matrix )
    TP = np.diag(confusion_matrix )
    TN = confusion_matrix .sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    SE = TP/(TP+FN)   # Sensitivity/ hit rate/ recall/ true positive rate
    SP = TN/(TN+FP)   # Specificity/ true negative rate  SP
    ACC_all=(TP+TN)/(FP+FN+TP+TN)
    return SE[label],SP[label],ACC_all[label],roc_auc_score(y_true,y_pred)