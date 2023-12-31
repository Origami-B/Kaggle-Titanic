# Kaggle-Titanic
The course project of Data Science, USTC, 2023Fall.


## 📄 项目简介 Description
2023年秋季学期数据科学导论大作业项目，选题为[Kaggle Titanic项目](https://www.kaggle.com/competitions/titanic/overview)，目标是预测乘客是否生还。



## 👤 项目成员 Members
* [宋林恺](https://github.com/)
* [杨城昊](https://github.com/)
* [罗胤玻](https://github.com/origami-b) 


## 📄 Kaggle Titanic项目说明

### 项目目标
通过训练集中的乘客信息，预测测试集中乘客是否生还。

### 数据集
数据集包含两个csv文件，分别为训练集和测试集以下逐一介绍：

#### 训练集 [train.csv](https://github.com/Origami-B/Kaggle-Titanic/blob/main/test.csv)
包含了891名乘客的信息，每个乘客有以下特征：
* PassengerId：乘客ID
* Survived：是否生还，0代表否，1代表是
* Pclass：船票等级，1代表一等舱，2代表二等舱，3代表三等舱
* Name：乘客姓名
* Sex：乘客性别
* Age：乘客年龄
* SibSp：乘客兄弟姐妹/配偶的数量
* Parch：乘客父母/孩子的数量
* Ticket：船票号码
* Fare：船票价格
* Cabin：船舱号码
* Embarked：登船港口，C代表Cherbourg，Q代表Queenstown，S代表Southampton


#### 测试集 [test.csv](https://github.com/Origami-B/Kaggle-Titanic/blob/main/test.csv)
包含了418名乘客的信息，每个乘客有以下特征：
* PassengerId：乘客ID
* Pclass：船票等级，1代表一等舱，2代表二等舱，3代表三等舱
* Name：乘客姓名
* Sex：乘客性别
* Age：乘客年龄
* SibSp：乘客兄弟姐妹/配偶的数量
* Parch：乘客父母/孩子的数量
* Ticket：船票号码
* Fare：船票价格
* Cabin：船舱号码
* Embarked：登船港口，C代表Cherbourg，Q代表Queenstown，S代表Southampton

### 提交要求
提交的结果文件应为一个csv文件，具体参考 [gender submssion.csv](https://github.com/Origami-B/Kaggle-Titanic/blob/main/gender_submission.csv) 

提交结果应包含两列，分别为PassengerId和Survived，其中PassengerId为测试集中乘客的ID，Survived为预测的结果，0代表不生还，1代表生还。

### 评价标准
将预测结果与线上测试集中的真实结果进行比较，计算预测正确的比例，即为准确率。

## 📅项目计划
* 熟悉项目内容，学习相关知识
* 数据预处理，特征工程
* 模型训练，调参
* 模型融合，提交结果

## 📅 项目进展 Progress Management
|    Date    |         Title         |                            Result                            |
| :--------: | :-------------------: | :----------------------------------------------------------: |
| 2023.10.19🌃 | 集体讨论选题 |      [Kaggle-Titanic](https://www.kaggle.com/competitions/titanic/overview)         |
| 2023.11.6🌃 | 前期调研 | 大致确定工作流程及注意事项  |
| 2023.11.11🌃 | 明确分工 | 罗胤玻-数据分析与预处理；宋林恺-特征选择与模型训练；杨城昊-模型调参 |
| 2023.11.12🌆 | 协作方式讨论 | 个人可使用jupyter-notebook，团队结果以python文件形式展示 |
| 2023.11.26🌃 | 数据可视化完成 | [data_visualization](./data_visualization.ipynb) | 
| 2023.12.3🌃 | 数据预处理完成 | [Titanic.py](./Titanic.py) | 
| 2023.12.6🌃 | 模型训练完成 | [Titanic.py](./Titanic.py) |              |
| 2023.12.7-12.12🌃 | 模型调参 |  |              |
| 2023.12.15🌃 | 实验报告完成 | [report](./report/report.md) |              |