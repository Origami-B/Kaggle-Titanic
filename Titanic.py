import featuretools as ft 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


train = pd.read_csv("./data/train.csv") 
test = pd.read_csv("./data/test.csv")

test["Survived"] = 0
combi = train._append(test, ignore_index=True)

combi.Embarked[combi.Embarked.isnull()] = combi.Embarked.dropna().mode().values
combi['Cabin'] = combi.Cabin.fillna('U0')
combi['Fare'] = combi[['Fare']].fillna(13.302889)

#choose training data to predict age
age_df = combi[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(combi['Age'].notnull())]
age_df_isnull = age_df.loc[(combi['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
combi.loc[combi['Age'].isnull(), ['Age']]= predictAges

# Sex: one-hot encoding
combi['Sex'] = pd.factorize(combi['Sex'])[0]
sex_dummies_df = pd.get_dummies(combi['Sex'], prefix=combi[['Sex']].columns[0])
combi = pd.concat([combi, sex_dummies_df], axis=1)

# Name: 不同称呼独热编码，增加名字长度特征 
combi['Title'] = combi['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

combi['Title'] = combi['Title'].map(title_Dict)

# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing
combi['Title'] = pd.factorize(combi['Title'])[0]

title_dummies_df = pd.get_dummies(combi['Title'], prefix=combi[['Title']].columns[0])
combi = pd.concat([combi, title_dummies_df], axis=1)

combi['Name_length'] = combi['Name'].apply(len)

# Pclass：各舱位按价格分类
# 建立PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

Pclass1_mean_fare = combi['Fare'].groupby(by=combi['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combi['Fare'].groupby(by=combi['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combi['Fare'].groupby(by=combi['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare Category
combi['Pclass_Fare_Category'] = combi.apply(pclass_fare_category, args=(
Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
pclass_level = LabelEncoder()

# 给每一项添加标签
pclass_level.fit(np.array(
['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))

# 转换成数值
combi['Pclass_Fare_Category'] = pclass_level.transform(combi['Pclass_Fare_Category'])

# dummy 转换
pclass_dummies_df = pd.get_dummies(combi['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
combi = pd.concat([combi, pclass_dummies_df], axis=1)

combi['Pclass'] = pd.factorize(combi['Pclass'])[0]

# Parch and SibSp: 合并为FamilySize，家庭成员数目
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

combi['Family_Size'] = combi['Parch'] + combi['SibSp'] + 1
combi['Family_Size_Category'] = combi['Family_Size'].map(family_size_category)

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combi['Family_Size_Category'] = le_family.transform(combi['Family_Size_Category'])

family_size_dummies_df = pd.get_dummies(combi['Family_Size_Category'],
                                     prefix=combi[['Family_Size_Category']].columns[0])
combi = pd.concat([combi, family_size_dummies_df], axis=1)

# Ticket: 提取数字，如果没有数字则为N，再factorize
combi['Ticket_Letter'] = combi['Ticket'].str.split().str[0]
combi['Ticket_Letter'] = combi['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)

# 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
# combi['Ticket_Number'] = combi['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
# combi['Ticket_Number'].fillna(0, inplace=True)

# 将 Ticket_Letter factorize
combi['Ticket_Letter'] = pd.factorize(combi['Ticket_Letter'])[0]

# Cabin: 有Cabin记录为Yes，缺失为No
combi.loc[combi.Cabin.isnull(), 'Cabin'] = 'U0'
combi['Cabin'] = combi['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

# Regulization
scale_age_fare = preprocessing.StandardScaler().fit(combi[['Age','Fare', 'Name_length']])
combi[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combi[['Age','Fare', 'Name_length']])

# 备份数据
combi_data_backup = combi
combi.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Pclass_Fare_Category', 
                       'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)

# 分割训练集与数据集
train_data = combi[:891]
test_data = combi[891:]

titanic_train_data_X = train_data.drop(['Survived'],axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'],axis=1)