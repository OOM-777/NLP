import zipfile
import os
import io
import random
import json
import matplotlib.pyplot as plt
import numpy as np
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
all_rumor_list = []
all_non_rumor_list = []

# 解析谣言数据
for rumor_class_dir in rumor_class_dirs:
    if (rumor_class_dir != '.DS_Store'):
        print(rumor_class_dir)
        # 遍历谣言数据，并解析
        with open(original_microblog + rumor_class_dir, 'r') as f:
            rumor_dict = json.load(f)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1

print(all_rumor_list)

# 解析非谣言数据
for non_rumor_class_dir in non_rumor_class_dirs:
    if (non_rumor_class_dir != '.DS_Store'):
        with open(original_microblog + non_rumor_class_dir, 'r') as f2:
            non_rumor_dict = json.load(f2)
        all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1

print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))

# 全部数据进行乱序后写入all_data.txt
data_list_path = "/home/aistudio/data/"
all_data_path = data_list_path + "all_data.txt"
all_data_list = all_rumor_list + all_non_rumor_list

random.shuffle(all_data_list)

# 在生成all_data.txt之前，首先将其清空
with open(all_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a') as f:
    for data in all_data_list:
        f.write(data)
print('all_data.txt已生成')



exit()






#
# data_path='./data/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/'
#
# data=pd.read_json('./data/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/1_z5qFIwiEj_2771041282.json')
# print(data)
# exit()
# # 观察发现编号为2601之前的全部是谣言
# # 0-2601是谣言；2601-5135是非谣言
#
# file_name=os.listdir(data_path)
# for i in range(len(file_name)):
#     file=data_path+file_name[i]
#
#     data=pd.read_json('./data/',lines=True)
#     print(data)
#     exit()
#
