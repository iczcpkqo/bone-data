import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # y.to_csv("out.txt", index=False, header=False)
    bankdata = pd.read_csv("C:/codedomain/database/traindata/bill_authentication.csv")
    # bankdata = pd.read_csv("C:/codedomain/database/traindata/testdata.txt", sep=" " ,header=None)
    # bankdata = pd.read_csv("C:/codedomain/database/traindata/testdata.txt", index=False, header=False)

    # print(bankdata.head())

    # X = bankdata.iloc[:,0:13]
    # y = bankdata.iloc[:,-1]
    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']
    # print(X)
    # print(y)
    # print(bankdata)

    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # print(X_train)
    # print(y_test)

    #
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    #
    y_pred = svclassifier.predict(X_test)
    #
    print(confusion_matrix(y_test,y_pred))
    # tt = confusion_matrix(y_test,y_pred)
    # print(tt)
    print(classification_report(y_test,y_pred))
    # tt = classification_report(y_test,y_pred)
    # print(tt)


if __name__ == "__main__":
    main()
