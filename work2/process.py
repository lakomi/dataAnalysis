# -*- coding:UTF-8 -*-
import os
import json
from association import Association
import pandas as pd

read_data = "data\\wine-reviews"
# data_file_list = ["winemag-data_first150k.csv"]
data_file_list = ["winemag-data_first150k.csv", "winemag-data-130k-v2.csv"]
write_data = "result_h2"


# 创建目录。
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 提取标称属性
def extract_nominal():
    for filename in data_file_list:
        content = pd.read_csv(os.path.join(read_data, filename))
        # \result_h2\数据集名\process_data.csv
        write_data_path = os.path.join(write_data, filename.split('.')[0])
        makedir(write_data_path)

        print("process file: ", filename)
        print("提取标称属性")
        for title in content.columns.values:
            if content[title].dtypes == "int64" or content[title].dtypes == "float64":
                content = content.drop(columns=[title])
        content = content.dropna()
        # print(content)
        print("转存为process_data.csv")
        with open(os.path.join(write_data_path,"process_data.csv"),'w',encoding="utf-8") as f:
            content.to_csv(f)


        print("转换为二元组")
        dataset = []
        for title in content.columns.values:
            data_set = []
            for value in content[title]:
                data_set.append((title, value))
            dataset.append(data_set)

        print("计算频繁项集")
        association = Association()
        # 调用apriori算法，得到频繁项集
        freq_set, support_data = association.apriori(dataset)
        support_data_out = sorted(support_data.items(), key= lambda d:d[1],reverse=True)
        print("频繁项集存储中。。。")
        # 将频繁项集结果，保存到文件中
        freq_set_file = open(os.path.join(write_data_path,"freq_set.json"),'w',encoding = "utf-8")
        for (key,value) in support_data_out:
            result_dict = {'set':None, 'sup':None}
            set_result = list(key)
            sup_result = value
            result_dict['set'] = set_result
            result_dict['sup'] = sup_result
            json_str = json.dumps(result_dict, ensure_ascii=False)
            freq_set_file.write(json_str+'\n')
        freq_set_file.close()

        print("获取关联规则列表")
        # 获取强关联规则列表
        big_rules_list = association.generate_rules(freq_set, support_data)
        big_rules_list = sorted(big_rules_list, key= lambda x:x[3], reverse=True)
        print(big_rules_list)
        print("关联规则存储中。。。")
        # 将关联规则输出到结果文件
        rules_file = open(os.path.join(write_data_path, 'rules.json'), 'w',encoding="utf-8")
        for result in big_rules_list:
            result_dict = {'X_set':None, 'Y_set':None, 'sup':None, 'conf':None, 'lift':None}
            X_set, Y_set, sup, conf, lift = result
            result_dict['X_set'] = list(X_set)
            result_dict['Y_set'] = list(Y_set)
            result_dict['sup'] = sup
            result_dict['conf'] = conf
            result_dict['lift'] = lift
            json_str = json.dumps(result_dict, ensure_ascii=False)
            rules_file.write(json_str + '\n')
        rules_file.close()
        print("当前文件结束"+'\n')

extract_nominal()
