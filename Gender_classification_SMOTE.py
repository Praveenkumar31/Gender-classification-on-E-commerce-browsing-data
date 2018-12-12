#import packages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier


#Load data from trainingData.csv and trainingLabels.csv
colnames_train = ['session_id', 'start_time', 'end_time', 'product_list']
colnames_test = ['label']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
df = pd.read_csv("./data/trainingData.csv", names=colnames_train, parse_dates=['start_time', 'end_time'], header=None)
df_test = pd.read_csv("./data/trainingLabels.csv", names = colnames_test)


#Feature_Engineering
df['session_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
df['product_count'] = df['product_list'].str.count('A')
df['Average_duration_per_product'] = df['session_duration'] / df['product_count']
df['day_of_week'] = df['end_time'].dt.dayofweek
df['date'] = df['end_time'].dt.day
df['month'] = df['end_time'].dt.month
df['start_hour'] = df['start_time'].dt.hour
df['end_hour'] = df['end_time'].dt.hour

#Scaling the product_count variable
sc = StandardScaler()
prd_cnt = np.array(df['product_count'])
prd_cnt = prd_cnt.reshape(-1, 1)
df['product_count'] = sc.fit_transform(prd_cnt)
df_dummy = df.iloc[:,7:12]


#dummy encoding the categorical data such as day_of_week, date, month, start_hour, end_hour
dummy = pd.get_dummies(df_dummy, columns=['day_of_week', 'date', 'month', 'start_hour', 'end_hour'])
df = pd.concat([df, dummy], axis = 1)


#preparing the data for identifying the number of nodes at each level/hierarchy
df['product_list'] = df.product_list.apply(lambda x: x.split('/'))
df['product_list'] = df.product_list.apply(' '.join).str.replace('[^A-Za-z0-9,\s]+', '').str.split(expand=False)
df['new'] = df['product_list'].map(set)
df['new'] = df['new'].apply(lambda x: ','.join(map(str, x)))


#calculating number of nodes at each level
df['A_unique_count'] = df['new'].str.count('A')
df['B_unique_count'] = df['new'].str.count('B')
df['C_unique_count'] = df['new'].str.count('C')
df['D_unique_count'] = df['new'].str.count('D')


#dropping the varibles which have been dummy-encoded
df.drop(['day_of_week', 'date', 'month', 'start_hour', 'end_hour', 'new'], inplace=True, axis=1)


#index of columns
print(df.columns)



#Dummy encoding products/hierarchical nodes using multilabel Binarizer
mlb = MultiLabelBinarizer()
df_product = pd.DataFrame(mlb.fit_transform(df['product_list']),columns=mlb.classes_, index=df.index)


#Dropping the products/hierarchical nodes that appear less than 3 times
df_product.drop([col for col, val in df_product.sum().iteritems() if val < 3], axis=1, inplace=True)


#Joining the features extracted with the dummy encoded product/hierarchical data. Dropped redundant and unwanted data for model construction.
df_product = pd.concat([df_product, df], axis = 1)
df_product.drop(['product_list', 'session_id', 'start_time', 'end_time'], inplace=True, axis=1)


#adding the target variable "gender"
df_product['gender'] = df_test['label']


#Converting female and male class to 0 and 1 respectively
labelencoder_y = LabelEncoder()
df_product['gender'] = labelencoder_y.fit_transform(df_product['gender'])


#Target varible - class proportion
df_product.groupby(['gender'])['day_of_week_1'].count()


#Creating X - feature and y - target. Train test split - 80:20
X = df_product.iloc[:,:-1].values
y = df_product.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Oversampling using SMOTE - Synthetic minority Oversampling technique
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)
print(X_train.shape)
print(y_train.shape)


#logistic Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Neural network
MLP_Classifier = MLPClassifier(random_state=4, learning_rate = 'invscaling', hidden_layer_sizes= (50,40,30))
MLP_Classifier.fit(X_train,y_train)
y_pred = MLP_Classifier.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Linear Support Vector classifier
Linear_SVC = LinearSVC()
Linear_SVC.fit(X_train,y_train)
y_pred = Linear_SVC.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Gradient Boosting Classifier
GBM = GradientBoostingClassifier()
# class_weight= {0:0.22, 1:0.78}
GBM.fit(X_train,y_train)
y_pred = GBM.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Random Forest Classifier
Randomforest = RandomForestClassifier(random_state= 134, n_estimators= 100)
Randomforest.fit(X_train,y_train)
y_pred = Randomforest.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Decision Tree Classifier
Decisiontree = DecisionTreeClassifier()
Decisiontree.fit(X_train,y_train)
y_pred = Decisiontree.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


#Ensemble using Voting classifier of Decision tree, Random forest, linear support vector classifier, Logistics Regression, Neural network, Gradient boosting classifier
ensemble = VotingClassifier(estimators = [('GBM', GBM), ('DT',Decisiontree), ('RF', Randomforest), ('SVC', Linear_SVC), ('NN', MLP_Classifier), ('LR', lr)], n_jobs=-1)
ensemble.fit(X_train,y_train)
y_pred = ensemble.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
score = ((tp/ (tp+fn)) + (tn/ (tn+fp)))/2
print("Score of the algorithm(computed as given in question):", score)
print("General accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))