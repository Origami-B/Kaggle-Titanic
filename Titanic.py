import numpy as np 
import pandas as pd 
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import seaborn as sns


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):

    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est =AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))
    
    # GradientBoosting
    gb_est =GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))
    
    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))
    
    # merge the three models
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 
                               ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 
                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    print('Sample 10 Features from all:')
    print(str(features_top_n[:20]))
    
    return features_top_n , features_importance

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


train = pd.read_csv("./data/train.csv") 
test = pd.read_csv("./data/test.csv")
PassengerId = test['PassengerId']

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

# feature_to_pick = 20
# feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
# titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
# titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

x_train = titanic_train_data_X.values # Creates an array of the train data
x_test = titanic_test_data_X.values # Creats an array of the test data
y_train = titanic_train_data_Y.values

rf = RandomForestClassifier(n_estimators=500, warm_start=False, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors = 6)

svm = SVC(kernel='linear', C=0.025)

rf.fit(x_train,y_train)
# rf_pred = rf.predict(x_test)
# rf_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': rf_pred})
# rf_sumb.to_csv('./data/rf_result.csv', index=False, sep=',')

ada.fit(x_train,y_train)
# ada_pred = ada.predict(x_test)
# ada_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': ada_pred})
# ada_sumb.to_csv('./data/ada_result.csv', index=False, sep=',')

et.fit(x_train,y_train)
# et_pred = et.predict(x_test)
# et_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': et_pred})
# et_sumb.to_csv('./data/et_result.csv', index=False, sep=',')

gb.fit(x_train,y_train)
# gb_pred = gb.predict(x_test)
# gb_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': gb_pred})
# gb_sumb.to_csv('./data/gb_result.csv', index=False, sep=',')

dt.fit(x_train,y_train)
# dt_pred = dt.predict(x_test)
# dt_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': dt_pred})
# dt_sumb.to_csv('./data/dt_result.csv', index=False, sep=',')

knn.fit(x_train, y_train)
# knn_pred = knn.predict(x_test)
# knn_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': knn_pred})
# knn_sumb.to_csv('./data/knn_result.csv', index=False, sep=',')

svm.fit(x_train,y_train)
# svm_pred = svm.predict(x_test)
# svm_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived': svm_pred})
# svm_sumb.to_csv('./data/svm_result.csv', index=False, sep=',')

# model = VotingClassifier(
#             estimators=[('rf', rf), ('ada', ada), ('et', et), ('gb', gb), ('dt', dt), ('knn', knn), ('svm', svm)],
#             voting='hard'
# )

# model.fit(x_train, y_train)
# voting_pred = model.predict(x_test)
# voting_sumb = pd.DataFrame({'PassengerId':PassengerId, 'Survived':voting_pred})
# voting_sumb.to_csv('./data/voting_result.csv', index=False, sep=',')

# Some useful parameters which will come in handy later on
ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, shuffle=False)

def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Create our OOF train and test predictions. These base results will be used as new features

rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost 
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector
x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('./data/Stack_result.csv',index=False,sep=',')
