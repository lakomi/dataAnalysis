import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# 两个数据集
data1_dir = "wine-reviews"
data1_file_list = ["winemag-data_first150k.csv", "winemag-data-130k-v2.csv"]

data2_dir = "oakland-crime-statistics-2011-to-2016"
data2_file_list = ["records-for-2011.csv", "records-for-2012.csv", "records-for-2013.csv",
                   "records-for-2014.csv", "records-for-2015.csv", "records-for-2016.csv"]

# 创建目录。
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class DataMining():
    def __init__(self):
        # 读取数据集的目录
        self.read_data_dir = [os.path.join(
            "data", data1_dir), os.path.join("data", data2_dir)]
        # 结果存放目录
        self.result_data_dir = [os.path.join(
            "result", data1_dir), os.path.join("result", data2_dir)]
        self.data_file_list = [data1_file_list, data2_file_list]

    def data_abstract(self):
        # 遍历每个数据集文件
        for i in range(len(self.data_file_list)):
            for file_name in self.data_file_list[i]:
                # 读取内容
                content = pd.read_csv(os.path.join(
                    self.read_data_dir[i], file_name))
                result_data_path = os.path.join(
                    self.result_data_dir[i], file_name.split('.')[0])

                print("当前正在处理的文件: ", file_name)
                for title in content.columns.values:
                    if title == "Unnamed: 0":
                        continue
                    # 数值
                    if content[title].dtypes == "int64" or content[title].dtypes == "float64":
                        self.process_num_features(
                            content, title, result_data_path)
                    else:
                        self.process_nom_features(
                            content, title, result_data_path)

    # 标称属性
    def process_nom_features(self, content, title, result_data_path):
        result_file_path = os.path.join(result_data_path, 'nominal_attribute')
        makedir(result_file_path)

        value_dict = self.get_feature_value(content, title)

        with open(os.path.join(result_file_path, title+".txt"), "w", encoding='utf-8') as fp:
            fp.write(("Feature Name: %s\n" % title))
            fp.write(("Value Num: %s\n" % len(value_dict)))
            for i in value_dict:
                fp.write(str(i) + "," + str(value_dict[i]) + "\n")

        print("标称属性")
        print("Feature Name: %s " % title)
        print("Value Num: %s" % len(value_dict))

    # 数值属性的统计
    def process_num_features(self, content, title, result_data_path):
        # 生成数值属性 结果目录
        result_file_path = os.path.join(result_data_path, 'numeric_attribute')
        makedir(result_file_path)

        # 五数概括
        min_num = content[title].min()
        quartile1 = content[title].quantile(0.25)
        median_num = content[title].median()
        quartile2 = content[title].quantile(0.75)
        max_num = content[title].max()
        # 缺失个数。count()一列非空格个数
        missing_num = len(content) - content[title].count()

        with open(os.path.join(result_file_path, title+".txt"), "w") as fp:
            fp.write(("Feature Name: %s\n" % title))
            fp.write(("Min Num: %s\n" % min_num))
            fp.write(("1/4 Quartile Num: %s\n" % quartile1))
            fp.write(("Median Num: %s\n" % median_num))
            fp.write(("3/4 Quartile Num: %s\n" % quartile2))
            fp.write(("Max Num: %s\n" % max_num))
            fp.write(("Missing Num: %s\n" % missing_num))

        print("数值属性统计")
        print("Feature Name: %s" % title)
        print("Min Num: %s" % min_num)
        print("1/4 Quartile Num: %s" % quartile1)
        print("Median Num: %s" % median_num)
        print("3/4 Quartile Num: %s" % quartile2)
        print("Max Num: %s" % max_num)
        print("Missing Num: %s" % missing_num)

        # 数据可视化
        result_figure_path = os.path.join(result_data_path, 'figure')
        makedir(result_figure_path)
        self.draw_figure(content, title, result_figure_path)

    def draw_figure(self, content, title, result_figure_path):
        # 直方图
        figure_path = os.path.join(result_figure_path, title + "_zhifang.png")
        self.draw_zhifang(content, title, figure_path)
        # 盒图
        figure_path = os.path.join(result_figure_path, title + "_box.png")
        self.draw_box(content, title, figure_path)

    def draw_zhifang(self, content, title, result_figure_path):
        # 获取非空行
        content = content.dropna(subset=[title])
        # 直方图
        plt.hist(content[title], 50)
        plt.title(title)
        plt.xlabel("value")
        plt.ylabel("frequence")
        plt.savefig(result_figure_path)
        plt.close()

    def draw_box(self, content, title, write_figure_path):
        content = content.dropna(subset=[title])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 画盒图。sym表示偏离值的表示方法
        ax.boxplot(content[title], labels=[title])
        plt.savefig(write_figure_path)
        plt.close()

    # 统计每个取值的频数
    def get_feature_value(self, content, title):
        value_dict = dict()
        for i in range(len(content)):
            # 是否为空
            if pd.isnull(content[title][i]):
                continue
            if content[title][i] in value_dict:
                value_dict[content[title][i]] += 1
            else:
                value_dict[content[title][i]] = 1
        return value_dict

    # 缺失处理方法一，删除缺失值
    def filling_method_1(self, content, result_data_path, file_name):
        method_path = os.path.join(result_data_path, "method_1")
        makedir(method_path)
        with open(os.path.join(method_path, file_name), 'w', encoding='utf-8') as fp1:
            method1_content = content
            for title in content.columns.values:
                if title == "Unnamed: 0":
                    continue
                elif method1_content[title].dtypes == "int64" or method1_content[title].dtypes == "float64":
                    # 丢弃其中缺失值的行
                    method1_content = method1_content.dropna(subset=[
                        title])
                    self.draw_figure(
                        method1_content, title, method_path)

    # 缺失处理方法二，用最高频率值来填补缺失值
    def filling_method_2(self, content, result_data_path, file_name):
        method_path = os.path.join(result_data_path, "method_2")
        makedir(method_path)
        with open(os.path.join(method_path, file_name), 'w', encoding='utf-8') as fp2:
            method2_content = content
            for title in content.columns.values:
                if title == "Unnamed: 0":
                    continue
                elif method2_content[title].dtypes == "int64" or method2_content[title].dtypes == "float64":
                    value_dict = self.get_feature_value(
                        method2_content, title)
                    filling_data = max(value_dict, key=value_dict.get)
                    # 用字典，填充不同的常数。
                    method2_content = method2_content.fillna(
                        {title: filling_data})
                    self.draw_figure(
                        method2_content, title, method_path)

    # 邻近点插值
    def filling_method_3(self, content, result_data_path, file_name):
        method_path = os.path.join(result_data_path, "method_3")
        makedir(method_path)
        with open(os.path.join(method_path, file_name), 'w', encoding='utf-8') as fp3:
            method3_content = content
            # 邻近点插值
            method3_content = method3_content.interpolate(method='nearest')
            for title in content.columns.values:
                if title == "Unnamed: 0":
                    continue
                elif method3_content[title].dtypes == "int64" or method3_content[title].dtypes == "float64":
                    self.draw_figure(
                        method3_content, title, method_path)

    # 方法四
    def filling_method_4(self, content, result_data_path, file_name):
        method_path = os.path.join(result_data_path, "method_4")
        makedir(method_path)
        with open(os.path.join(method_path, file_name), 'w', encoding='utf-8') as fp4:
            method4_content = content
            nonan_content = pd.DataFrame()
            num_list = []
            for title in method4_content.columns.values:
                if title == "Unnamed: 0":
                    method4_content = method4_content.drop(title, 1)
                elif method4_content[title].dtypes == "int64" or method4_content[title].dtypes == "float64":
                    num_list.append(title)
                    nonan_content = pd.concat(
                        [nonan_content, method4_content[title]], axis=1)
            # axis=0 删除有缺失值的行，any表示该行只要有缺失值就删。
            nonan_content.dropna(axis=0, how='any', inplace=True)
            mean_val = [nonan_content[title].mean()
                        for title in nonan_content.columns.values]

            if len([title for title in num_list if method4_content[title].isnull().any() == True]) == len(num_list):
                for i in range(len(method4_content)):
                    if method4_content.loc[i][num_list].isnull().all():
                        for j in range(len(num_list)):
                            method4_content.loc[i, num_list[j]] = mean_val[j]

            for title in num_list:
                if method4_content[title].isnull().any():
                    train_y1 = nonan_content[title]
                    train_x1 = nonan_content.loc[:, [
                        other for other in num_list if other != title]]
                    test_x1 = method4_content[pd.isna(method4_content[title])].loc[:, [
                        other for other in num_list if other != title]]
                    index, pred = self.knn_filled(
                        train_x1, train_y1, test_x1)
                    method4_content.loc[index, title] = pred
                self.draw_figure(method4_content, title, method_path)

    def filling(self):
        for name in range(len(self.data_file_list)):
            for file_name in self.data_file_list[name]:
                content = pd.read_csv(os.path.join(
                    self.read_data_dir[name], file_name))
                result_data_path = os.path.join(
                    self.result_data_dir[name], file_name.split('.')[0])

                print("缺失处理。当前处理文件: %s" % file_name)
                print("策略1处理")
                self.filling_method_1(content, result_data_path, file_name)
                print("策略2处理")
                self.filling_method_2(content, result_data_path, file_name)
                print("策略3处理")
                self.filling_method_3(content, result_data_path, file_name)
                print("策略4处理")
                self.filling_method_4(content, result_data_path, file_name)

    def knn_filled(self, x_train, y_train, test, k=4, dispersed=True):
        if dispersed:
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        else:
            clf = KNeighborsRegressor(n_neighbors=k, weights="distance")

        clf.fit(x_train, y_train)
        return test.index, clf.predict(test)


if __name__ == "__main__":
    data = DataMining()

    data.data_abstract()

    data.filling()
