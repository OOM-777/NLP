import pandas as pd
import joblib
import lstm_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
enc = preprocessing.OneHotEncoder()


TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

# 用于存放精准率、召回率等指标
FPR = []
TPR = []
AUC = []

# 绘制混淆矩阵
def plt_confusion_matrix(PATH, Y_TEST, Y_PRE, TITLE):
    sns.set()
    f, ax = plt.subplots()
    y_true = Y_TEST
    y_pred = Y_PRE
    C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(C2)
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(C2, annot=True, ax=ax, xticklabels=['Not—Rumor ', 'Rumor'], yticklabels=['Not—Rumor ', 'Rumor'],
                cmap='Blues')  # 画热力图
    ax.set_title(TITLE)  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig(PATH)
    plt.show()


# 神经网络绘制loss图像
def show(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# 训练贝叶斯模型
def bayers(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print('======================开始Bayes模型训练:======================')
    # 训练保存模型
    nb = BernoulliNB()
    model = nb.fit(X_TRAIN, Y_TRAIN)  # 训练模型
    joblib.dump(filename=BAYERS_PATH, value=model)

    # 开始预测
    y_predict = model.predict(X_TEST)

    # 计算精准率、召回率、F1分数
    precision = precision_score(Y_TEST, y_predict)
    recall = recall_score(Y_TEST, y_predict)
    f1mean = f1_score(Y_TEST, y_predict)

    print('精确率为：%0.5f' % precision)
    print('召回率：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)
    path = './picture/bayers.png'
    title = 'Bayers Confusion Matrix'
    # 绘制混淆矩阵
    plt_confusion_matrix(path, Y_TEST, y_predict, title)
    # 计算AUC和ROC
    fpr, tpr, thresholds_keras = roc_curve(Y_TEST, y_predict)
    Auc = auc(fpr, tpr)
    print("AUC : ", Auc)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(Auc)


# 训练KNN模型
def KNN(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print("======================开始训练KNN模型======================")
    knn = KNeighborsClassifier(n_neighbors=2)
    model = knn.fit(X_TRAIN, Y_TRAIN)

    joblib.dump(filename=KNN_PATH, value=model)

    y_predict = model.predict(X_TEST)
    print(type(Y_TEST), type(y_predict))

    precision = precision_score(Y_TEST, y_predict)
    recall = recall_score(Y_TEST, y_predict)
    f1mean = f1_score(Y_TEST, y_predict)

    print('精确率为：%0.5f' % precision)
    print('召回率为：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)

    path1 = './picture/knn.png'
    path2 = './picture/knn_roc.png'
    title1 = 'Knn Confusion Matrix'
    title2 = 'Knn ROC curve'

    # 绘制混淆矩阵
    plt_confusion_matrix(path1, Y_TEST, y_predict, title1)
    # 计算AUC和ROC
    fpr, tpr, thresholds_keras = roc_curve(Y_TEST, y_predict)
    Auc = auc(fpr, tpr)
    print("AUC : ", Auc)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(Auc)
    return y_predict, Y_TEST

# 决策树
def decision_tree(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print("======================开始训练决策树模型======================")
    tree = DecisionTreeClassifier(criterion='gini', max_depth=19, random_state=3)
    model = tree.fit(X_TRAIN, Y_TRAIN)
    joblib.dump(filename=DECSION_TREE_PATH, value=model)

    y_predict = model.predict(X_TEST)

    precision = precision_score(Y_TEST, y_predict)
    recall = recall_score(Y_TEST, y_predict)
    f1mean = f1_score(Y_TEST, y_predict)

    print('精确率为：%0.5f' % precision)
    print('召回率为：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)

    path1 = './picture/decision_tree.png'
    path2 = './picture/decision_tree_rpc.png'
    title1 = 'Decision tree Confusion Matrix'
    title2 = 'Decision tree ROC curve'
    # 绘制混淆矩阵
    plt_confusion_matrix(path1, Y_TEST, y_predict, title1)
    # 计算AUC和ROC
    fpr, tpr, thresholds_keras = roc_curve(Y_TEST, y_predict)
    Auc = auc(fpr, tpr)
    print("AUC : ", Auc)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(Auc)

# 随机森林
def random_forest(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print("======================开始训练随机森林模型======================")
    # 决策树数量、划分节点、最大深度
    forest = RandomForestClassifier(n_estimators=190, criterion='entropy', max_depth=25)
    model = forest.fit(X_TRAIN, Y_TRAIN)
    joblib.dump(filename=RANDOM_FOREST_PATH, value=model)

    y_predict = model.predict(X_TEST)

    precision = precision_score(Y_TEST, y_predict)
    recall = recall_score(Y_TEST, y_predict)
    f1mean = f1_score(Y_TEST, y_predict)

    print('精确率为：%0.5f' % precision)
    print('召回率为：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)

    path = './picture/random_forest.png'
    title = 'Random forest Confusion Matrix'
    # 绘制混淆矩阵
    plt_confusion_matrix(path, Y_TEST, y_predict, title)
    # 计算AUC和ROC
    fpr, tpr, thresholds_keras = roc_curve(Y_TEST, y_predict)
    Auc = auc(fpr, tpr)
    print("AUC : ", Auc)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(Auc)

# 支持向量机
def SVM(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print("======================开始训练SVM模型======================")
    svm = SVC(C=5, kernel='linear')  # 惩罚参数、核函数
    model = svm.fit(X_TRAIN, Y_TRAIN)
    joblib.dump(filename=SVM_PATH, value=model)

    y_predict = model.predict(X_TEST)

    precision = precision_score(Y_TEST, y_predict)
    recall = recall_score(Y_TEST, y_predict)
    f1mean = f1_score(Y_TEST, y_predict)

    print('精确率为：%0.5f' % precision)
    print('召回率为：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)

    path = './picture/svm.png'
    title = 'Svm Confusion Matrix'
    # 绘制混淆矩阵
    plt_confusion_matrix(path, Y_TEST, y_predict, title)
    # 计算AUC和ROC
    fpr, tpr, thresholds_keras = roc_curve(Y_TEST, y_predict)
    Auc = auc(fpr, tpr)
    print("AUC : ", Auc)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(Auc)


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
# 构造机器学习分类器训练测试集
x_train = train.iloc[:, :40]
y_train = train.iloc[:, [40]]

x_test = test.iloc[:, :40]
y_test = test.iloc[:, [40]]

print(x_test.shape, x_train.shape)

# # 构造深度学习分类器训练测试集
# l_x_train = np.array(x_train)
# l_x_train = l_x_train.reshape(2709, 1, 40)
#
# l_x_test = np.array(x_test)
# l_x_test = l_x_test.reshape(678, 1, 40)
#
#
# l_x_train, l_x_val, y_train, y_val = train_test_split(l_x_train, y_train, test_size=0.2, random_state=19)
#
#
# Model=lstm_model.LSTM_Attention()
# model=Model.make_model(l_x_train)
# history=model.fit(l_x_train,y_train,epochs=400,batch_size=64,validation_data=(l_x_val, y_val))
# model.save('./model/new_cnn.h5')
# pre = model.predict(l_x_test)
#
#
# y_pre = []
# for i in range(len(pre)):
#     temp = list(pre[i])
#     temp = temp.index(max(temp))
#     y_pre.append(temp)
#
# precision = precision_score(y_test, y_pre)
# recall = recall_score(y_test, y_pre)
# f1 = f1_score(y_test, y_pre)
# print('精确率为：%0.5f' % precision)
# print('召回率：%0.5f' % recall)
# print('F1均值为：%0.5f' % f1)
#
# show(history)


# KNN(x_train, y_train, x_test, y_test)


# decision_tree(x_train, y_train, x_test, y_test)
#
# bayers(x_train, y_train, x_test, y_test)
# #
# SVM(x_train, y_train, x_test, y_test)
# #
# random_forest(x_train, y_train, x_test, y_test)
#
#
