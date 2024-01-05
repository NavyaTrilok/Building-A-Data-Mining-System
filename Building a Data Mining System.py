#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt',header=None)
df


# In[3]:


import pandas as pd
df = pd.read_csv('D:/Masters/1.Assignments/IFT 511/Extra Credit lab/googleplaystore.csv')
df


# In[4]:


print(df.isnull().any())


# In[5]:


df.isnull().sum()


# In[6]:


for c in df.columns: 
     print( df[c].isnull().sum() )


# In[7]:


df.drop(['Rating'],axis=1).columns


# In[8]:


df.dropna()


# In[10]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)


# In[11]:




StratifiedKFold(n_splits=4, random_state=None, shuffle=False)
>>> for train_index, test_index in skf.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]


# In[2]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# In[ ]:





# In[1]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train, y_train)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [10,50,100,20000]
}
  

DT = DecisionTreeClassifier()
pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               
for k,train, test in enumerate(kfold):
    pipeline.fit(X_train_d.iloc[train, :], y_train_d.iloc[train])
    score = pipeline.score(X_train_d.iloc[test, :], y_train_d.iloc[test])
    scores.append(score)
    print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[17]:


directions = ['turn right', 'turn left', 'go straight', 'turn right']

for step, direction in enumerate(directions, start=1):
    print(f'Step {step}: {direction}')


# In[44]:


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd



print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

# Create an instance of Pipeline
#
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
#
# Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
#
strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train_d, y_train)
scores = []
#
#
#
for k, (train, test) in enumerate(kfold):
    pipeline.fit(X_train_d.loc[train, :], y_train_d.loc[test])
    score = pipeline.score(X_train_d.loc[train, :], y_train_d.loc[test])
    scores.append(score)
    print(kfold)
    print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))


# In[ ]:





# In[17]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train, y_train)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [10,50,100,20000]
}
  

DT = DecisionTreeClassifier()
pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               
for k,(train, test) in enumerate(kfold):
    scores.append(score)
    print(kfold)
 

                         
                         


# In[16]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [10,50,100,20000]
}
  

pipe_steps = [('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)
                         
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


for train_index, test_index in kf.split(X_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    


# In[28]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}
  

DT = tree.DecisionTreeClassifier()

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               
for train, test in strtfdKFold.split(X_train_d, y_train):
    DT.fit(X_train_d, y_train)
    score = DT.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
   
   


# In[11]:


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os


print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 15/colon-cancer_01.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
#
# Create an instance of Pipeline
#
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
#
# Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
#
strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train, y_train)
scores = []
#
#
#
for k, (train, test) in enumerate(kfold):
    print(k,train,test)


# In[26]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}
  

DT = tree.DecisionTreeClassifier()

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               
for train, test in strtfdKFold.split(X_train_d, y_train):
    print(train,test)


# In[29]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}
  

DT = tree.DecisionTreeClassifier()

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())

DT.fit([ 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
 33 34 35 36 37 38 39 40 41 42] [0 1 2 3 4 5 6 8 9])
DT.Score([ 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
 33 34 35 36 37 38 39 40 41 42] [0 1 2 3 4 5 6 8 9])
scores.append(score)
    print('Accuracy: %.3f' % (score))
                                               
#for train, test in strtfdKFold.split(X_train_d, y_train):
#    DT.fit(X_train_d, y_train)
#    score = DT.score(X_train_d, y_train)
#    scores.append(score)
#    print('Accuracy: %.3f' % (score))
 
#print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
   
   


# In[30]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

SS = StandardScaler()
DT = DecisionTreeClassifier()

pipe_steps = [('scaler',StandardScaler()),('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

print(pipe)
param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [5,10,15,20]
}

search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_d, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# In[32]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}
  
SS = StandardScaler()
DT = tree.DecisionTreeClassifier()

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())

pipe_steps = [('scaler',StandardScaler()),('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)


                                               
for train, test in strtfdKFold.split(X_train_d, y_train):
    pipe.fit(X_train_d, y_train)
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X_train_d, y_train)
    score = pipe.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
   
   


# In[54]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

  

DT = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2000)

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               for train, test in strtfdKFold.split(X_train_d, y_train):
    DT.fit(X_train_d, y_train)
    score = DT.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

DT2 = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2000)

for train, test in strtfdKFold.split(X_train_d, y_train):
    DT2.fit(X_train_d, y_train)
    score = DT2.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


pipe_steps = [('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

print(pipe)
param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
   


# In[43]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

  

DT = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2000)

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())
                                               
for train, test in strtfdKFold.split(X_train_d, y_train):
    DT.fit(X_train_d, y_train)
    score = DT.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

DT2 = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2000)

for train, test in strtfdKFold.split(X_train_d, y_train):
    DT2.fit(X_train_d, y_train)
    score = DT2.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
   

SS = StandardScaler()
DT = DecisionTreeClassifier()

pipe_steps = [('scaler',StandardScaler()),('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
   


# In[50]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
kfold = strtfdKFold.split(X_train_d, y_train)
scores = []

param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}
  

DT = tree.DecisionTreeClassifier()

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())

                                               
for train, test in enumerate(kfold):
    print(train,test)


# In[52]:


import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")
X_d = X.todense()

SS = StandardScaler()
DT = DecisionTreeClassifier()

pipe_steps = [('scaler',StandardScaler()),('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

print(pipe)
param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# In[53]:


import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Extra Credit lab/colon-cancer.txt")
X_d = X.todense()


DT = DecisionTreeClassifier()

pipe_steps = [('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

print(pipe)
param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# In[58]:


from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as npy

print("Loading Dataset...")
X,y = load_svmlight_file("D:/Masters/1.Assignments/IFT 511/Lab 9/a1a.txt")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train_d = X_train.todense()
X_test_d = X_test.todense()

strtfdKFold = StratifiedKFold(n_splits=5)
#kfold = strtfdKFold.split(X_train_d, y_train_d)
scores = []

  

DT = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2000)

#pipeline = make_pipeline(('descT', DecisionTreeClassifier())
for train, test in strtfdKFold.split(X_train_d, y_train):
    DT.fit(X_train_d, y_train)
    score = DT.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

DT2 = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2000)

for train, test in strtfdKFold.split(X_train_d, y_train):
    DT2.fit(X_train_d, y_train)
    score = DT2.score(X_train_d, y_train)
    scores.append(score)
    print('Accuracy: %.3f' % (score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


pipe_steps = [('descT', DecisionTreeClassifier())]
pipe = Pipeline(pipe_steps)

print(pipe)
param_grid = {
    "descT__criterion" :['gini', 'entropy'],
    "descT__max_depth": [20000]
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train_d, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
   


# In[ ]:




