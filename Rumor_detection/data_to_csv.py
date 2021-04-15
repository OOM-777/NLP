import os
import json
import jieba
import pandas as pd

target_path = "./data/Chinese_Rumor_Dataset-master"

# 分别为谣言数据、非谣言数据、全部数据的文件路径
rumor_class_dirs = os.listdir(target_path + "/CED_Dataset/rumor-repost/")

non_rumor_class_dirs = os.listdir(target_path + "/CED_Dataset/non-rumor-repost/")
original_microblog = target_path + "/CED_Dataset/original-microblog/"
# 谣言标签为0，非谣言标签为1
rumor_label = "0"
non_rumor_label = "1"

# 分别统计谣言数据与非谣言数据的总数
rumor_num = 0
non_rumor_num = 0

# 用于存放全部微博内容，方便后边制造停用词表
txt_data=str()
data = pd.DataFrame(columns=['label', 'content'])
# 解析谣言数据
for rumor_class_dir in rumor_class_dirs:
    if rumor_class_dir != '.DS_Store':
        # 遍历谣言数据，并解析
        with open(original_microblog + rumor_class_dir, 'r') as f:
            rumor_dict = json.load(f)

        #        test['label']=rumor_label
        txt_data=txt_data+str(rumor_dict["text"])
        data = data.append([{'content': rumor_dict["text"], 'label': rumor_label}], ignore_index=True)
        # all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1

# 解析非谣言数据
for non_rumor_class_dir in non_rumor_class_dirs:
    if non_rumor_class_dir != '.DS_Store':
        with open(original_microblog + non_rumor_class_dir, 'r') as f2:
            non_rumor_dict = json.load(f2)
        txt_data=txt_data+non_rumor_dict["text"]
        data = data.append([{'content': non_rumor_dict["text"], 'label': non_rumor_label}], ignore_index=True)
        # all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1



print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))

print(data)

# 导出生成csv文件
#data.to_csv('./data/data_to_csv.csv')

stop_data=jieba.lcut(txt_data)
print(type(stop_data))
count={}
for word in stop_data:
    count[word]=count.get(word,0)+1


# 查看出现次数最高的字符
print(sorted(count.items(),key=lambda i:i[1],reverse=True))

stop=[]
for key,values in count.items():

    if values>=225:
        stop.append(key)

print(stop)
