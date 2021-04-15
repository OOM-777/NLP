import pandas as pd
import jieba

data=pd.read_csv('./data/data_to_csv.csv')

stop=['【', '】','：', '5', '元', '"','！', '"', '"',' ', '你', '？', '】', '《', '》', '，', '上', '的', '都', '。', '是', '到', '、', '能', '了', '最', '要', '还', '人', '孩子', '10', '这', '在', '中国', '让', '个', '“', '”', '他们', '给', ',', '又', '我们', '1', '但', '（', '）', '没有', '我', ':', '年', '月', '日', '请', '多', '转发', '为', '@', '被', '她', '!', '也', '后', '将', '和', '去', '看', '他', '分', '吧', '[', ']', '就', '对', '有', '一', '着', '一个', '就是', '7', '-', '如果', '可以', '这个', '而', '朋友', '大家', '中', '#', 'http', '/', 't', '.', 'cn', '自己', '时', '很', '不', '~', '转', '岁', '—', '3', '好', '说', '啊', '会', '…', '把', '2', '爱', '与', '4', '来', '；', '6']

jieba_data=pd.DataFrame(columns=['after_jieba','label'])

for index,row in data.iterrows():
    #print(row['content'])
    words=jieba.lcut(row['content'])
    for word in words:
        if word in stop:
            words.remove(word)

    jieba_data=jieba_data.append([{'after_jieba':words,'label':row['label']}],ignore_index=True)

print(jieba_data)

jieba_data.to_csv('./data/jieba_data.csv')


