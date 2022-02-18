import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

credit_card_data = pd.read_csv('creditcard.csv')

credit_card_data.head()

#Checking the number of missing values
credit_card_data.isnull().sum()

credit_card_data['Class'].value_counts()

legit  = credit_card_data[credit_card_data['Class']==0]
fraud = credit_card_data[credit_card_data['Class']==1]

print(legit.shape)
print(fraud.shape)

# legit.Amount.describe()
# fraud.Amount.describe()

credit_card_data.groupby('Class').mean()

egit_sample = legit.sample(n=492)

#Concatenating  two dataframes
new_dataset = pd.concat([legit_sample,fraud],axis=0)
new_dataset.head()

new_dataset.groupby('Class').mean()

credit_card_data.groupby('Class').mean()

X = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

#Accuracy score on training data
train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(y_train, train_prediction)
print("Accuracy score on Training data : ",train_data_accuracy)

#Accuracy score on test data
test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, test_prediction)

print("Accuracy score on Test data : ",test_data_accuracy)



















