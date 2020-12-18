import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import numba as nb
import sys

TRAIN_DATA = "./train_data/"
RESULT_DATA_PATH = "./result_data/"
RESULT_REPORT_PATH = "./result_report/"
RESULT_GRAPH_PATH = "./result_graph/"
MODEL_PATH = "./trained_model/"
# OUT_DATA_PATH = "./out_data/"
PRESENT_TIME = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))


# MAIN , REAL MAIN
def run():
    model_name = '0.5_SV_Sigmoid__2020-12-18 00.59.51.m'
    # threshold = 0.5
    is_probility = True

    # Function Menu
    is_draw_roc = True
    is_write_data = True

    # for big_C in np.arange(0.1, 1.05, 0.1):
    print(model_name)

    y_test, y_test_probility, y_pred, test_pred = run_model(model_name, is_probility)

    # 绘制ROC
    if is_draw_roc:
        draw_roc(model_name , y_test, y_test_probility)

    # 保存数据
    if is_write_data:
        write_data(model_name, y_test, y_pred, test_pred)


def run_model(model_name, is_probility):
    # 数据
    ## tiny data
    url = TRAIN_DATA + "train-io-tiny.txt"
    pre_url = TRAIN_DATA + "test-in-tiny.txt"
    ## big data
    # url = "./tain-data/tain-io.txt"
    # pre_url = "./tain-data/test-in.txt"

    # Assign colum names to the dataset
    colnames = ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12', 'class']
    pre_colnames = ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12']

    # Read dataset to pandas dataframe
    train_data= pd.read_csv(url, names=colnames, sep=' ')
    pre_data = pd.read_csv(pre_url, names=pre_colnames, sep=' ')

    # 预处理
    X = train_data.drop('class', axis=1)
    y = train_data['class']

    # 分离数据。训练数据和验证数据。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # 调用模型
    svclassifier = joblib.load(MODEL_PATH + model_name)

    # 绘制ROC
    ## 获得验证数据的得分
    y_test_probility = svclassifier.decision_function(X_test)
    # print(y_test_probility)

    # 预测。预测验证数据和测试数据的得分。
    ## 验证数据
    y_pred = svclassifier.predict(X_test)
    ## 测试数据
    test_pred = svclassifier.predict(pre_data)

    # 大量输出数据
    np.set_printoptions(threshold=sys.maxsize)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(y_pred)

    # if is_sel == 0:

    return y_test, y_test_probility, y_pred, test_pred

# 输出文件
def write_data(model_name, y_test, y_pred, test_pred):
    # 保存验证数据产生的报告，反应模型情况
    file = open(RESULT_REPORT_PATH + '[' + model_name + ']_Report__' + PRESENT_TIME + '.txt', 'w')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n')
    file.write(str(classification_report(y_test, y_pred)))
    file.write('\n')
    file.write(str(y_pred))
    file.close()

    # 保存使用模型将测试数据分类结果
    file = open(RESULT_DATA_PATH + '[' + model_name + ']_Out__' + PRESENT_TIME + '.txt', 'w')
    file.write(str(test_pred))
    file.close()


# 绘图, ROC
def draw_roc(model_name, y_test, y_test_probility):
    # 得分。获得验证数据的得分。
    fpr, tpr, threshold = roc_curve(y_test, y_test_probility)

    # 绘制
    plt.ion()  # 开启interactive mode 成功的关键函数
    fig_name = RESULT_GRAPH_PATH + '[' + model_name + ']_ROC__' + PRESENT_TIME + '.png'
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

