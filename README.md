# Starbucks_Capstone_Project

### 项目背景与动机

这个数据集是一些模拟 Starbucks rewards 移动 app 上用户行为的数据。每隔几天，星巴克会向 app 的用户发送一些推送。这个推送可能仅仅是一条饮品的广告或者是折扣券或 BOGO（买一送一）。顾客收到的推送可能是不同的。

我们会将交易数据、人口统计数据和推送数据结合起来判断哪一类人群会受到某种推送的影响。这个数据集是从星巴克 app 的真实数据简化而来。

这个项目可以建立一个机器学习模型，来预测顾客是否会响应报价。 这也可以用于根据人口统计预测支出性质。 这项研究是Udacity拥有的课程的一部分。

### 依赖的运行环境

Python的Anaconda发行版中的常见库即可运行代码，如Pandas Numpy seaborn sklearn tqdm等。详见Jupyter notebook文件。

### 文档说明

- 在Jupyter notebook文件`Starbucks_Capstone_notebook-zh.ipynb`中主要包含两部分内容，
    1. 数据读取、预处理
    2. 机器学习模型
- 需要读取的原始数据，即在 ./data 路径下的.json 文件

### 成果输出

#### 1. 机器学习模型的输出

即 `./models_0219 路径下的.joblib文件`

#### 2. 清洗后数据集的输出

将 DataFrame中 clean_data 输出为 `.db 数据库文件`

#### 3. Report

分析报告，即 `Report.md`

### 改进方向

- 由于客户收入、报价持续时间、报价奖励这些数字特征对于模型影响较大：可以创建一个融合以上特征的全新多项式指标来改善机器学习模型。如new_feature特征，`new_feature = duration*reward/income`，之后将该特征也做归一化处理，进行模型训练。
- 可以尝试用主成分分析，去减少数据集的feature维度。
