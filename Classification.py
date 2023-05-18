from sklearn import metrics
from SSLM import *
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import numpy.random as rn
import cvxopt
from math import *


def show_message_for_data(X, Y, SSLM, dataset='Train'):

    f_vec = np.array([SSLM.predict(X[:, i], i) for i in range(X.shape[1])])
    f_vec = np.sign(f_vec)
    y_mat = f_vec == Y
    print('\r\n')
    print('**** show_message_for_data  ****')
    print(' data.dataset    ', dataset)
    print(' data.N          ', X.shape[1])

    print(' accuracy         %.3f%%  ' % ((np.sum(y_mat != 0) / X.shape[1]) * 100))
    print('\r\n')



if __name__ == "__main__":
    # Use the extracted features as input for the SSLM classifier

    rn.seed(100)



    # Train SSLM Classifier
    rbf_kernel_custom = lambda *args: rbf_kernel(*args, gamma=10.0)
    #kern = linear_kernel
    kern = rbf_kernel_custom
    sslm = SSLM(X_train, y_train, kern, nu=1.0, nu1=0.01, nu2=0.01, proba=True)
    show_message_for_data(X_test.T, y_test.T, sslm, dataset="Test")

    def f(x):
        tmp = sslm.predict_proba(x)
        return tmp, tmp

    y_train_pred = np.array([sslm.predict(X_train[:, i], i) for i in range(X_train.shape[1])])
    y_train_pred = np.sign(y_train_pred)
    print(" k - train_test  F-score: {0:.2f}".format(f1_score(y_train_pred, y_test,average='weighted')))

    y_test_pred = np.array([sslm.predict(X_test.T[:, i], i) for i in range(X_test.T.shape[1])])
    y_test_pred = np.sign(y_test_pred)

    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    cm = confusion_matrix(y_test,  y_test_pred)
    print('cm is:\n', cm)

    confusion = metrics.confusion_matrix(y_test, y_test_pred)

    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    TP = confusion[1, 1]
    print(TN, FP, FN, TP)

    # 混淆矩阵指标
    # 准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('accuracy:', accuracy)
    # 精确率：预测结果为正的样本中，预测正确的比例
    precision = TP / (TP + FP)
    print('precision:', precision)
    # 灵敏度(召回率：正样本中，预测正确的比例)
    recall = TP / (TP + FN)
    print('recall:', recall)
    # 特异度：负样本中，预测正确的比例
    specificity = TN / (TN + FP)
    print('specificity:', specificity)
    # F1分数：综合Precision和Recall的一个判断指标
    f1_recall = 2 * precision * recall / (precision + recall)
    print('f1:', f1_recall)
    # G-mean:不平衡分类评价指标
    Gmean = sqrt(recall * specificity)
    print('Gmean:', Gmean)
