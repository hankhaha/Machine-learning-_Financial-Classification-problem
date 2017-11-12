# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 07:40:48 2017

@author: Chun Wei Lo
"""

#%%Import Data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#%%
df= pd.read_csv("C:/Users/Chun Wei Lo/Desktop/WalletHub/Statistician Dataset.csv")
#%%
# check the class distributions
# count the number of instances in response variables
from collections import Counter
df.y.describe()
no=Counter(df["y"])[0]
yes=Counter(df["y"])[1]
print(yes/no)
sns.countplot("y",data=df)
#%%
# remove outlier 
df_n=df[df.y<=2]
sns.countplot("y",data=df_n)
#%%
# Specify predictors and target
# Predictors
df_pred=df_n.drop("y",axis=1)
# Target
df_target=df_n.loc[:,"y"]
#%% Feature Selection-RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
#%%
# use linear regression as the model
lr = LogisticRegression()
names=df_pred.columns
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(df_pred,df_target)

#%%
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))


import numpy as np
select_RFE=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
RFE_select=pd.DataFrame(select_RFE)
old_names=[0,1]
new_names=["Rank","Features"]
RFE_select.rename(columns=dict(zip(old_names, new_names)), inplace=True)
RFE_select.head(40)
#%%
a=range(0,48)
important_feature=RFE_select.iloc[a,:]
#%%
# splitting to training data and test data 
from sklearn.model_selection import train_test_split
# Select 80% of data as train set and 20% of data as test
predictors=important_feature.Features.values.ravel()
target_feature=["y"]
X=df_n.loc[:,predictors]
x_train,x_test,y_train,y_test=train_test_split(X,df_n.y,test_size=0.20,random_state=7)

#%%#
#%% cross-validation

kf_10=KFold(x_train.shape[0],n_folds=10,shuffle = True, random_state = 12345)
#%%
# Create a function for Parameters Tuning  
from sklearn.model_selection import GridSearchCV
def Classification_model_GridSearch(model,params):
    clf = GridSearchCV(model,params,scoring= 'accuracy',cv=kf_10)
    clf.fit(x_train,y_train.values.ravel())
    print("best score is :")
    print(clf.best_score_)
    print('')
    print("best estimator is :")
    print(clf.best_estimator_)

    return (clf.best_score_)

#%%
# Final evaluation for models
def Classification_model_accuracy(model):
    model.fit(x_train,y_train.values.ravel())
    pred=model.predict(x_test)
    #pred_prob=model.predict_proba(x_test)
    accuracy=accuracy_score(y_test,pred)
    roc=roc_auc_score(y_test,pred)
    
    #CM=confusion_matrix(test_y,pred)
    #roc_score=roc_score(test_y,pred)
    #CR=classification_report(test_y,pred)
    #pr=precision_score(test_y,pred)
    return accuracy,roc
#%%SVM with parameter tuning
svm_param_grid = {'C': [3,2,1,0.5,0.1,0.01,0.001],
                  'cache_size':[200,100,300]}
svm_model = SVC(random_state=12345)
Classification_model_GridSearch(svm_model,svm_param_grid)
#%% Best SVM
Best_SVM=SVC(C=3, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=12345, shrinking=True,
  tol=0.001, verbose=False)
#%% Logistics Regression with parameter tuning
lr_param_grid = {'C': [3,2,1,0.5,0.1,0.05,0.01,0.001]}
lr_model = LogisticRegression(random_state=12345)
Classification_model_GridSearch(lr_model,lr_param_grid)

#%% Best Logistics
Best_lr_model=LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=12345, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

#%%Random Forest with parameter tuning
rf_param_grid = {'min_samples_leaf': [1,3,5,7,9],
                  'min_samples_split':[2,3,4,5],
                  'n_estimators':[100,200,300,50,25,10]}
rf_model = RandomForestClassifier(n_estimators = 100,random_state=12345)
Classification_model_GridSearch(rf_model,rf_param_grid)
#%% Best rf model
Best_rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=3, min_weight_fraction_leaf=0.0,
            n_estimators=25, n_jobs=1, oob_score=False, random_state=12345,
            verbose=0, warm_start=False)

#%%
# Boosting
# Gradiant Boosting Decision Tree
import pandas
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
Boosing=GradientBoostingClassifier(n_estimators=20,random_state=12345)
#%%
# Widely used Machine learning Models 
models=["SVM","Logistic_Regression","Random Forest","Gradient Boosting"]
# The arguement in the models can be modified, such as 100 n_estimators or number of n_neighbors
Classification_models =[Best_SVM,Best_lr_model,Best_rf,Boosing]
#%%
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=3, min_weight_fraction_leaf=0.0,
            n_estimators=25, n_jobs=1, oob_score=False, random_state=12345,
            verbose=0, warm_start=False)

#%% Model comparision
results = []
scoring = 'accuracy'
for model in Classification_models:
	cv_results = model_selection.cross_val_score(model,x_train,np.ravel(y_train), cv=kf_10, scoring=scoring)
	results.append(cv_results)
#%%
# boxplot algorithm comparison
fig = plt.figure(figsize=(8,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models)
plt.show()
#%%
# Evaluate performance of original ML models 
Accuracy=[]
AUC_ROC=[]
seed=np.random.seed(1)
for model in Classification_models:
    accuracy=Classification_model_accuracy(model)[0]
    roc=Classification_model_accuracy(model)[1]
    Accuracy.append(accuracy)
    AUC_ROC.append(roc)
#%%

# Create a dataframe for new ML model
Precision_with_model = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with important features":Accuracy,
     "AUC_ROC with important features":AUC_ROC
    })
    #%%