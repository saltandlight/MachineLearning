from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('train data shape:', train.shape)
print('test data shape:', test.shape)
print('----------[train information]----------')
print(train.info())
print('----------[test information]--------')
print(test.info())

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    print(feature_ratio)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i+1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df = df.fillna(0)
    print(df)
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()

train_and_test = [train, test]
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady'
                                                 , 'Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)

for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)

print(train.head())
print(train.Embarked.value_counts(dropna=False))

for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)

# Age Feature
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
print(train_and_test)

for dataset in train_and_test:
   dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
   dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
   dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
   dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
   dataset.loc[dataset['Age'] > 64, 'Age'] = 4
   dataset['Age'] = dataset['Age'].map({0: 'Child', 1: 'Young', 2:'Middle', 3: 'Prime', 4: 'Old'}).astype(str)

# Fare Feature
print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])

for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675)

for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

# SibSp & Parch Feature (Family)
for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)

# 특성 추출 및 나머지 전처리
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand'], axis=1)

print(train.head())
print(test.head())

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop('PassengerId', axis=1).copy()

train_data, train_label = shuffle(train_data, train_label, random_state=5)

def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy:", accuracy, "%")
    return prediction

# Logistic Regression
log_pred = train_and_test(LogisticRegression(max_iter=4000))
# SVM
svm_pred = train_and_test(SVC())
# KNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors=4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Naive Bayes
nb_pred = train_and_test(GaussianNB())
