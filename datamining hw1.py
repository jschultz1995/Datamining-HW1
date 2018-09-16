# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

pd.options.display.max_columns = 12

def header(msg) :
    print('-' * 50)
    print(msg)
    

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combine = [train_df, test_df]

header("Feature Names:")
print(combine[0].columns.tolist())

header("Categorical Features:")
print(combine[0].select_dtypes(include=['object']).columns.tolist())

header("Numerical Features:")
print(combine[0].select_dtypes(include=[np.number]).columns.tolist())

header("Features NaN Values:")
print(combine[0].columns[combine[0].isnull().any()].tolist())

header("Data Types:")
print(combine[0].dtypes)

header("Properties of Numerical Features:")
print(combine[0][combine[0].columns[3:]].select_dtypes(include=[np.number]).describe())

header("Properties of Categorical Features:")
combine[0]['PassengerId'] = combine[0].PassengerId.astype('object')
combine[0]['Survived'] = combine[0].Survived.astype('object')
combine[0]['Pclass'] = combine[0].Pclass.astype('object')
print(combine[0].select_dtypes(include=['object']).describe())

header("Correlation of Pclass = 1 and Survived")
combine[0]['Survived'] = combine[0].Survived.astype(int)
combine[0]['Pclass'] = combine[0].Pclass.astype(int)

print(train_df["Survived"][train_df["Pclass"] == 1].value_counts(normalize = True))

header("Gender Survival Rates:")
header("Female")
print(train_df["Survived"][train_df["Sex"] == "female"].value_counts(normalize = True))
header("Male")
print(train_df["Survived"][train_df["Sex"] == "male"].value_counts(normalize = True))
#print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

age_hist = train_df.loc[:, ['Survived', 'Age']]
age_hist.hist(column = 'Age',by='Survived', bins = 20)
plt.savefig('image.png')

three_hist = train_df.loc[:, ['Survived', 'Age', 'Pclass']]
three_hist.hist(column = 'Age', by=['Survived','Pclass'],sharex = True, bins = 15)
plt.savefig('image2.png')


survived_df = train_df.loc[train_df['Survived'] == 1]
not_survived_df = train_df.loc[train_df['Survived'] == 0]

sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Fare", col='Embarked',
                    data=survived_df, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Fare")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survived');
plt.savefig('image3.png')

sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Fare", col='Embarked',
                    data=not_survived_df, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Fare")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Did Not Survive');
plt.savefig('image4.png')

header("Null valuse in Cabin Feature:")
print(train_df['Cabin'].isnull().sum() + test_df['Cabin'].isnull().sum())

header("Replacing sex values with 0's or 1's:")
train_df['Sex'].replace(['female','male'],[1,0], inplace=True)
train_df.rename(columns={'Sex' : 'Gender'}, inplace=True)
print(train_df['Gender'].head())

header("Filling Age Values")
def fillNaN_with_unifrand(df):
    a = df.values
    m = np.isnan(a) # mask of NaNs
    mu, sigma = df.mean(), df.std()
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df

fillNaN_with_unifrand(train_df['Age'])
print(train_df['Age'])

header("Filling Embarked Values")
train_df['Embarked'].fillna(value='S' ,inplace=True)

header("Getting Fare's Mode")
print(train_df['Fare'].mode())

header("Converting Fare to Ordinal")
train_df.loc[train_df['Fare'].between(-0.001, 7.91), 'Fare'] = 0
train_df.loc[train_df['Fare'].between(7.910000000001,14.454), 'Fare'] = 1
train_df.loc[train_df['Fare'].between(14.45400000001,31.0), 'Fare'] = 2
train_df.loc[train_df['Fare'].between(31.00000000001,512.329), 'Fare'] = 3
print(train_df.head())
