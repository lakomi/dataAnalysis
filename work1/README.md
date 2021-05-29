# 数据探索性分析与数据预处理
## 数据集选择
        Oakland Crime Statistics 2011 to 2016
        Wine Reviews
## 运行环境及相关依赖
        python 3.7
        pandas、matplotlib、sklearn
## 运行方法
        python main.py
        
        如果要运行，首先要在main.py的同级目录下，新建一个data目录，将数据集放入其中，再执行上述命令
## 文件结构
        dataAnalysis/
         main.py
         --result                         # 结果目录
           --wine-reviews
             --winemag-data_first150k
               --figure                   # 图片结果目录
                 --*_zhifang.png
                 --*_box.png
               --nominal_attribute        # 标称属性结果目录
                 --*.txt
               --numeric_attribute        # 数值属性结果目录
                 --*.txt
               --method_1                 # 缺失值处理方法1结果目录
                 --winemag-data_first150k.csv
                 --*_zhifang.png
                 --*_box.png
               --method_2                 # 缺失值处理方法2结果目录
                 --...
             --winemag-data-130k-v2
               --...
          --oakland-crime-statistics-2011-to-2016
            --...
