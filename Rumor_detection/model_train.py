import pandas as pd
import joblib
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
from sklearn.linear_model import LogisticRegression

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
    sns.heatmap(C2, annot=True, ax=ax, xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], cmap='Blues')  # 画热力图
    ax.set_title(TITLE)  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig(PATH)
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

# 决策树
def decision_tree(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    print("======================开始训练决策树模型======================")
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=19, random_state=3)
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

x_train = train.iloc[:, :40]
y_train = train.iloc[:, [40]]

x_test = test.iloc[:, :40]
y_test = test.iloc[:, [40]]
print(x_train.head())
print(y_train.head())

KNN(x_train, y_train, x_test, y_test)
#
decision_tree(x_train, y_train, x_test, y_test)
#
bayers(x_train, y_train, x_test, y_test)
#
SVM(x_train, y_train, x_test, y_test)
#
random_forest(x_train, y_train, x_test, y_test)