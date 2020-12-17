import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

# 预处理
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

# 分离数据

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# 选分类器
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

# 做预测
y_pred = svclassifier.predict(X_test)

# 评估算法
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))