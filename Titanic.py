import featuretools as ft 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("./data/train.csv") 
test = pd.read_csv("./data/test.csv")

test["Survived"] = 0
combi = train._append(test, ignore_index=True)

combi.Embarked[combi.Embarked.isnull()] = combi.Embarked.dropna().mode().values
combi['Cabin'] = combi.Cabin.fillna('U0')
combi['Fare'] = combi[['Fare']].fillna(13.302889)
from sklearn.ensemble import RandomForestRegressor

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