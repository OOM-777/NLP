
# 基于机器学习的微博中文谣言分析
## 数据集
    数据集详见（Rumor_detection/data/Chinese_Rumor_Dataset-master/README.md）
    此项目仅涉及第二部分微博原文数据
    
## 数据处理
    对微博原文进行分词，得到的词语去除停用词后放入TF-IDF模型中进行训练，之后进行SVD降维
    模型见Rumor_detection/model
## 建模训练
    使用常见的机器学习模型，学习效果如下：
    ![Alt text](https://github.com/OOM-777/NLP/blob/master/Rumor_detection/picture/knn.png)

    详见Rumor_detection/pirture
    
## 集成学习
    由多个若学习器通过加权投票的方法集成强学习器进行谣言检测
