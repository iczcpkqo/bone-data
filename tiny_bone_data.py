
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import numba as nb
import sys

def run():
    s_sel = 2
    big_C = 0.5
    # threshold = 0.5
    is_probility = True

    # Function Menu
    is_draw_roc = True

    # for big_C in np.arange(0.1, 1.05, 0.1):
    print(big_C)

    y_test, y_test_probility = run_case(s_sel, big_C, is_probility)

    # 绘制ROC
    if is_draw_roc:
        draw_roc(y_test, y_test_probility)


# for i in range(2, 3):
    #     run_case(i,big_C)

def run_case(is_sel, big_C, is_probility):
    # 数据
    ## tiny data
    url = "C:/codedomain/database/traindata/train-io-tiny.txt"
    pre_url = "C:/codedomain/database/traindata/test-in-tiny.txt"
    ## big data
    # url = "C:/codedomain/database/traindata/train-io.txt"
    # pre_url = "C:/codedomain/database/traindata/test-in.txt"

    # Assign colum names to the dataset
    colnames = ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12', 'class']
    pre_colnames = ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12']

    # Read dataset to pandas dataframe
    train_data= pd.read_csv(url, names=colnames, sep=' ')
    pre_data = pd.read_csv(pre_url, names=pre_colnames, sep=' ')

    # 预处理
    X = train_data.drop('class', axis=1)
    y = train_data['class']

    # 分离数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # 选分类器
    ## 高斯
    if is_sel == 0:
        print('Gau')
        svclassifier = SVC(kernel='rbf', probability=is_probility, C=big_C)

    ## 多项式
    if is_sel == 1:
        print('Pol')
        svclassifier = SVC(kernel='poly', probability=is_probility, degree=8, C=big_C)
        svclassifier.fit(X_train, y_train)

    ## sigmoid
    if is_sel == 2:
        print('Sig')
        svclassifier = SVC(kernel='sigmoid', probability=is_probility, C=big_C)

    # 训练
    svclassifier.fit(X_train, y_train)

    # 绘制ROC
    ## 获得得分
    y_test_probility = svclassifier.decision_function(X_test)
    # print(y_test_probility)

    ## 获得真假率
    fpr, tpr, threshold = roc_curve(y_test, y_test_probility)

    # 预测评估
    y_pred = svclassifier.predict(X_test)
    test_pred = svclassifier.predict(pre_data)

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('max_colwidth', 10000000)
    np.set_printoptions(threshold=sys.maxsize)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(y_pred)

    name_report_file = ''
    name__file= ''
    if is_sel == 0:
        file = open(str(big_C) + '_Gaussin_Report.txt', 'w')
        file.write(str(confusion_matrix(y_test, y_pred)))
        file.write('\n')
        file.write(str(classification_report(y_test, y_pred)))
        file.write('\n')
        file.write(str(y_pred))
        file.close()

        file = open(str(big_C) + '_Gaussin_Out.txt', 'w')
        file.write(str(test_pred))
        file.close()

    elif is_sel == 1:
        file = open(str(big_C) + '_Polynomial_Report.txt', 'a')
        file.write(str(confusion_matrix(y_test, y_pred)))
        file.write('\n')
        file.write(str(classification_report(y_test, y_pred)))
        file.write('\n')
        file.write(str(y_pred))
        file.close()

        file = open(str(big_C) + '_Polynomial_Out.txt', 'w')
        file.write(str(test_pred))
        file.close()

    elif is_sel == 2:
        file = open(str(big_C) + '_Sigmoid_Report.txt', 'a')
        file.write(str(confusion_matrix(y_test, y_pred)))
        file.write('\n')
        file.write(str(classification_report(y_test, y_pred)))
        file.write('\n')
        file.write(str(y_pred))
        file.close()

        file = open(str(big_C) + '_Sigmoid_Out.txt', 'w')
        file.write(str(test_pred))
        file.close()
    return y_test, y_test_probility

## 绘图, ROC
def draw_roc(y_test, y_test_probility):
    # 获得真假率
    fpr, tpr, threshold = roc_curve(y_test, y_test_probility)

    # 绘制
    plt.ion()  # 开启interactive mode 成功的关键函数
    fig_name = "let's play ROC__" + str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())) + '.png'
    plt.figure(fig_name, figsize=(6, 6))

    # average time of each eat
    plt.subplot(1, 1, 1)
    plt.title("avg arrive time")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr,tpr , '-g', lw=1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    # plt.legend(loc="lower right")
    plt.savefig(fig_name)



    # plt.pause(1.001)
    # clear memory
    # plt.clf()  # clear

# for i in range(0,8):
run()
