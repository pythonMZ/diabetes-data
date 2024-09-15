# diabetes-data


libraries :
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly as ply
import math
import warnings
warnings.filterwarnings('ignore')

#libraries use fir model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import ttest_ind,ttest_1samp
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  SVC , NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    recall_score,precision_score,
    precision_recall_fscore_support,
    precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


X=df.drop('Diabetes_012',axis=1)
Y=df['Diabetes_012']

x_train,x_temp,y_train,y_temp=train_test_split(X,Y,stratify=Y,random_state=42,test_size=0.2)

x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,random_state=42,test_size=0.2)

x_train.shape[0],x_val.shape[0],x_test.shape[0]

y_train.value_counts()

"""**SCALE**"""

scaler=StandardScaler()

x_train_scale=scaler.fit_transform(x_train)

x_val_scale=scaler.transform(x_val)

"""**PCA**"""

pca=PCA(n_components=0.9)

x_train_pca=pca.fit_transform(x_train_scale)

x_val_pca=pca.transform(x_val_scale)

"""**RandomForest**"""

#RandomForestClassifier
forest_model=RandomForestClassifier(random_state=42,n_estimators=150)
forest_model.fit(x_train_scale,y_train)

forest_y_pred=forest_model.predict(x_val)

forst_class=classification_report(y_val,forest_y_pred)
print(forst_class)

forest_recall=recall_score(y_val,forest_y_pred)
forest_recall

forest_f1=f1_score(y_val,forest_y_pred)
 forest_f1

forest_precision=precision_score(y_val,forest_y_pred)
forest_precision

forest_accuracy=accuracy_score(y_val,forest_y_pred)
forest_accuracy

forest_confusion=confusion_matrix(y_val,forest_y_pred)
sns.heatmap(forest_confusion,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

importances = forest_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.show()

"""**Oversampling**"""

#Oversampling(SMOTE)
smote=SMOTE(random_state=42)
x_over_train,y_over_train=smote.fit_resample(x_train,y_train)

x_over_train.shape[0]

y_over_train.value_counts()

x_over_train_scale=scaler.transform(x_over_train)
x_over_train_pca=pca.transform(x_over_train_scale)
forest_model=RandomForestClassifier(random_state=42,n_estimators=150)
forest_model.fit(x_over_train_pca,y_over_train)

forest_y_pred=forest_model.predict(x_val_pca)

forst_class=classification_report(y_val,forest_y_pred)
print(forst_class)

forest_recall=recall_score(y_val,forest_y_pred)
forest_recall

forest_f1=f1_score(y_val,forest_y_pred)
 forest_f1

forest_precision=precision_score(y_val,forest_y_pred)
forest_precision

forest_accuracy=accuracy_score(y_val,forest_y_pred)
forest_accuracy

forest_confusion=confusion_matrix(y_val,forest_y_pred)
sns.heatmap(forest_confusion,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

"""**Undeersampling**"""

#Undersampling
under_sample=RandomUnderSampler(random_state=42)
x_under_train,y_under_train=under_sample.fit_resample(x_train,y_train)

x_under_train.shape[0]

y_under_train.value_counts()

x_under_train_scale=scaler.transform(x_under_train)
x_under_train_pca=pca.transform(x_under_train_scale)
forest_model=RandomForestClassifier(random_state=42,n_estimators=150)
forest_model.fit(x_under_train_pca,y_under_train)

forest_y_pred=forest_model.predict(x_val_pca)

forst_class=classification_report(y_val,forest_y_pred)
print(forst_class)

forest_recall=recall_score(y_val,forest_y_pred)
forest_recall

forest_f1=f1_score(y_val,forest_y_pred)
 forest_f1

forest_precision=precision_score(y_val,forest_y_pred)
forest_precision

forest_accuracy=accuracy_score(y_val,forest_y_pred)
forest_accuracy

forest_confusion=confusion_matrix(y_val,forest_y_pred)
sns.heatmap(forest_confusion,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

"""**SVD**"""

svc=SVC()
svc.fit(x_under_train_pca,y_under_train)

y_pred_SVC=svc.predict(x_val_pca)

SVC_class=classification_report(y_val,y_pred_SVC)
print(SVC_class)

SVC_recall=recall_score(y_val,y_pred_SVC)
SVC_recall

SVC_f1=f1_score(y_val,y_pred_SVC)
SVC_f1

SVC_precision=precision_score(y_val,y_pred_SVC)
SVC_precision

SVC_accuracy=accuracy_score(y_val,y_pred_SVC)
SVC_accuracy

SVC_confusion=confusion_matrix(y_val,y_pred_SVC)
sns.heatmap(SVC_confusion,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

"""**VotingClassifier**"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

model_lr = LogisticRegression()#Logistic Regression
model_dt = DecisionTreeClassifier()#Decision Tree
model3 = SVC(probability=True)#SVD best parametr

voting_clf = VotingClassifier(estimators=[('lr', model_lr),('dt', model_dt),('svc', model3)],voting='soft')

voting_clf.fit(x_train,y_train)

predictions = voting_clf.predict(x_val)

v_class=classification_report(y_val,predictions)
print(v_class)

v_recall=recall_score(y_val,predictions)
v_recall

v_f1=f1_score(y_val,predictions)
v_f1

v_precision=precision_score(y_val,predictions)
vprecision

v_accuracy=accuracy_score(y_val,predictions)
v_accuracy

v_confusion=confusion_matrix(y_val,predictions)
sns.heatmap(v_confusion,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
