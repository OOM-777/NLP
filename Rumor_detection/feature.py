import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from config import TF_IDF_PATH,FEATURE_PATH,SVD_PATH,TRAIN_PATH,TEST_PATH


df = pd.read_csv('./data/jieba_data.csv', sep=',')
df.dropna(axis=0, how='any', inplace=True)  # 按行删除空缺行

print(df)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(df[['after_jieba']], df['label'], test_size=0.3,
                                                    random_state=0)
print("训练数据集大小：%d" % x_train.shape[0])
print("测试集数据大小：%d" % x_test.shape[0])

# 转换为DataFrame
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


# 重置索引，方便后面合并表格,因为此时label和数据的索引是乱序的，后面合并表格的时候对不上号
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 取出邮件内容部分
x_train = list(x_train['after_jieba'].astype('str'))
print("====================================处理前的数据====================================")
#print(x_train[199])
#exit()
print("正在进行特征工程，请耐心等待~")

# TF-IDF
vectorizer = CountVectorizer(decode_error="replace")
tfidftransformer = TfidfTransformer()
conut_train = vectorizer.fit_transform(x_train)
tfidf = tfidftransformer.fit_transform(conut_train)

# 测试集进行TF—IDF转换
content_test = list(x_test['after_jieba'].astype('str'))
count_test = vectorizer.transform(content_test)
data_test = tfidftransformer.transform(count_test)

# 保存TF—IDF文件
with open(FEATURE_PATH, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)
with open(TF_IDF_PATH, 'wb') as fw:
    pickle.dump(tfidftransformer, fw)

# SVD降维
# 多次实验得到降至20维效果最好
svd = TruncatedSVD(n_components=40)
svd_model = svd.fit(tfidf)

df2 = svd_model.transform(tfidf)
data = pd.DataFrame(df2)

data_test = svd_model.transform(data_test)
data_test = pd.DataFrame(data_test)

# 保存SVD模型
joblib.dump(filename=SVD_PATH, value=svd_model)


# 将训练集和测试集的数据和标签进行合并（按照列方向进行合并）
new_data = pd.concat([data, y_train], axis=1)
new_test = pd.concat([data_test, y_test], axis=1)

# 导出成csv文件
new_data.to_csv(TRAIN_PATH, index=False)
new_test.to_csv(TEST_PATH, index=False)