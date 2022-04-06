#!/usr/bin/env python
# coding: utf-8

# In[7]:


np.__version__


# In[ ]:





# In[8]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv("adult.csv")


# In[10]:


data.head()


# In[11]:


data.shape


# In[12]:


data.info()


# Summary of the dataset shows that there are no missing values. But the preview shows that the dataset contains values coded as `?`. So, I will encode `?` as NaN values.

# #### Encode ? as NaNs

# In[13]:


data[data == " ?"] = np.nan


# Again check the summary of dataframe

# In[14]:


data.info()


# In[15]:


data.isnull().sum()


# Now, the summary shows that the variables - workclass, occupation and country contain missing values. All of these variables are categorical data type. So, I will impute the missing values with the most frequent value- the mode.

# #### Impute missing values with mode

# In[16]:


for col in ["workclass", "occupation", "country"]:
  data[col].fillna(data[col].mode()[0], inplace=True)


# In[17]:


data.isnull().sum()


# In[18]:


salary = {' <=50K' : 0, ' >50K' : 1}
data["salary"] = data["salary"].map(salary)


# ### Exploratory Data Analysis

# #### Numerical variables

# In[19]:


# list of numerical variables
numerical_features = [feature for feature in data.columns if data[feature].dtypes != "O"]
print("Number of numerical variables: ", len(numerical_features))

# Visualize the numerical variables
data[numerical_features].head()


# Numerical variables are usually of 2 types:
# 1. Continous variable 
# 2. Discrete Variables

# ##### Discrete Variables

# In[20]:


discrete_variables = [feature for feature in numerical_features if len(data[feature].unique()) < 25]
print("LNumber of discrete variables is: ", len(discrete_variables))


# In[21]:


discrete_variables


# ##### Continous variables

# In[22]:


continuous_variables = [feature for feature in numerical_features if feature not in discrete_variables]
print("Continuous feature count: {}".format(len(continuous_variables)))


# In[23]:


# Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_variables:
  dataset = data.copy()
  dataset[feature].hist(bins=25)
  plt.xlabel(feature)
  plt.ylabel("Count")
  plt.title(feature)
  plt.show()


# #### Categorical variables

# In[24]:


categorical_features = [feature for feature in data.columns if data[feature].dtypes == "O" ]

print("Length of categorical features is: ", len(categorical_features))


# In[25]:


categorical_features


# In[26]:


data[categorical_features].head()


# In[27]:


for feature in categorical_features:
  print("The feature in {} and number of categories are {}".format(feature, len(data[feature].unique())))


# #### Missing values

# In[33]:


# Here we will check the nan values in each features.

features_with_na = [feature for feature in data.columns if data[feature].isnull().sum() > 1 ]
features_with_na


# Since, in there are no null values 

# #### Correlation

# In[34]:


corrmat = data.corr()
plt.figure(figsize=(10,10))
# Plot heat map
g = sns.heatmap(corrmat, annot=True)


# In[35]:


# Explore Education Num vs Salary
p = sns.factorplot(x = "education-num", y = "salary", data = data, kind = "bar", size = 6, palette = "muted")
p.despine(left=True)
p = p.set_ylabels(">50K probability")


# In[36]:


# Explore Hours Per Week vs Salary
p = sns.factorplot(x = "hours-per-week", y = "salary", data = data, kind = "bar", size = 6, palette = "muted")
p.despine(left=True)
p = p.set_ylabels(">50K probability")


# In[37]:


# Explore Age vs Salary
p = sns.FacetGrid(data, col = "salary")
p = p.map(sns.distplot, "age")
plt.show()


# In[38]:


# Explore Sex vs Salary
p = sns.barplot(x = "sex", y= "salary", data = data)
p = p.set_ylabel("Income >50K Probability")
plt.show()


# In[27]:


# Explore Relationship vs Salary
p = sns.factorplot(x = "relationship", y = "salary", data = data, kind = "bar", size = 6, palette = "muted")
p.despine(left = True)
p = p.set_ylabels("Income >50K Probability")
plt.show()


# In[39]:


# Explore Marital Status vs Income
p = sns.factorplot(x = "marital-status", y = "salary", data = data, kind = "bar", size = 6, palette = "muted")
p.despine(left = True)
p = p.set_ylabels("Income >50K Probability")
plt.show()


# In[40]:


# Explore Workclass vs Salary
p = sns.factorplot(x = "workclass", y = "salary", data = data, kind = "bar", size = 6, palette = "muted")
p.despine(left = True)
p = p.set_ylabels("Income >50K Probability")
plt.show()


# ### Feature Engineering

# ##### One-hot Encoding

# In[41]:


data["sex"] = pd.get_dummies(data["sex"], drop_first=True)


# In[42]:


data.head()


# ##### Label encoding

# In[43]:


from sklearn.preprocessing import LabelEncoder
categories = ["workclass", "education","occupation"]
for feature in categories:
  le = LabelEncoder()
  data[feature] = le.fit_transform(data[feature])


# In[44]:


data.head()


# In[45]:


categorical_data = ["marital-status", "race", "relationship", "country"]
for i in categorical_data:
  print(data[i].unique())


# In[46]:


maritalstatus = {' Never-married' : 1, ' Married-civ-spouse' : 2, ' Divorced' : 3, ' Married-spouse-absent' : 4, 
                 ' Separated' : 5, ' Married-AF-spouse' : 6, ' Widowed' : 8 }

race = {' White' : 5, ' Black' : 4, ' Asian-Pac-Islander' : 3, ' Amer-Indian-Eskimo' : 2, ' Other' : 1}

relationship = {' Not-in-family' : 1, ' Husband' : 2, ' Wife' : 3, ' Own-child' : 4, ' Unmarried' : 5,
 ' Other-relative' : 6}

country = {' United-States' : 1, ' Cuba' : 2, ' Jamaica' : 3, ' India' : 4, ' Mexico' : 5, ' South' : 6,
 ' Puerto-Rico' : 7, ' Honduras' : 8, ' England' : 9, ' Canada' : 10, ' Germany' : 11, ' Iran' : 12,
 ' Philippines' : 13, ' Italy' : 14, ' Poland' : 15, ' Columbia' : 16, ' Cambodia' : 17, ' Thailand' : 18,
 ' Ecuador' : 19, ' Laos' : 20, ' Taiwan' : 21, ' Haiti' : 22, ' Portugal' : 23, ' Dominican-Republic' : 24,
 ' El-Salvador' : 25, ' France' : 26, ' Guatemala' : 27, ' China' : 28, ' Japan' : 29, ' Yugoslavia' : 30,
 ' Peru' : 31, ' Outlying-US(Guam-USVI-etc)' : 32, ' Scotland' : 33, ' Trinadad&Tobago' : 34,
 ' Greece' : 35, ' Nicaragua' : 36, ' Vietnam' : 37, ' Hong' : 38, ' Ireland' : 39, ' Hungary' : 40,
 ' Holand-Netherlands' : 41}


# In[47]:


data["marital-status"] = data["marital-status"].map(maritalstatus)
data["race"] = data["race"].map(race)
data["relationship"] = data["relationship"].map(relationship)
data["country"] = data["country"].map(country)


# In[48]:


data.head()


# In[49]:


data.drop(["fnlwgt", "education-num"], axis = 1, inplace = True)


# In[50]:


data.head()


# In[51]:


data["salary"].value_counts()


# In[52]:


sns.countplot(data["salary"], label = "Count")
plt.show()


# #### Setting feature vector and target variable

# In[53]:


X = data.drop(["salary"], axis = 1)
y = data["salary"]


# In[54]:


X.head()


# #### Split data into separate training and test set

# In[55]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[56]:


print(len(X_train))


# In[57]:


print(len(X_test))


# #### Feature Scaling

# In[58]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[59]:


X_train.head()


# In[60]:


get_ipython().system('pip install imbalanced-learn')


# In[61]:


from collections import Counter
from imblearn.over_sampling import SMOTE


# In[62]:


sm=SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_res)))


# ### Model Creation

# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve


# ##### Logistic Regression

# In[65]:


log = LogisticRegression(C = 0.5, max_iter = 500)
log.fit(X_train_res, y_train_res)


# In[66]:


y_pred = log.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[67]:


metrics.plot_roc_curve(log, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred, average= None)


# ##### Random Forest Classifier

# In[68]:


rf = RandomForestClassifier(n_estimators = 200)
rf.fit(X_train_res, y_train_res)


# In[69]:


y_pred1 = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred1))
print(accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))


# In[70]:


metrics.plot_roc_curve(rf, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred1, average= None)


# ##### xgboost classifier

# In[71]:


xgb = XGBClassifier(learning_rate = 0.35, n_estimator = 500)
xgb.fit(X_train_res, y_train_res)


# In[72]:


y_pred2 = xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))


# In[73]:


metrics.plot_roc_curve(xgb, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred2, average= None)


# ##### Gaussian Naine Bayes

# In[74]:


gnb = GaussianNB()
gnb.fit(X_train_res, y_train_res)


# In[75]:


y_pred3 = gnb.predict(X_test)
print(confusion_matrix(y_test, y_pred3))
print(accuracy_score(y_test, y_pred3))
print(classification_report(y_test, y_pred3))


# In[76]:


metrics.plot_roc_curve(gnb, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred3, average= None)


# ##### Support Vector Classifier

# In[77]:


svc = SVC(kernel = 'rbf', max_iter = 1000, probability = True)
svc.fit(X_train_res, y_train_res)


# In[78]:


y_pred4 = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred4))
print(accuracy_score(y_test, y_pred4))
print(classification_report(y_test, y_pred4))


# In[79]:


metrics.plot_roc_curve(svc, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred4, average= None)


# ##### K Nearest Neighbors Classifier

# In[80]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_res, y_train_res)


# In[81]:


y_pred5 = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred5))
print(accuracy_score(y_test, y_pred5))
print(classification_report(y_test, y_pred5))


# In[82]:


metrics.plot_roc_curve(knn, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred5, average= None)


# ##### Decision Tree Classifier

# In[83]:


Dtree = DecisionTreeClassifier()
Dtree.fit(X_train_res, y_train_res)


# In[84]:


y_pred6 = Dtree.predict(X_test)
print(confusion_matrix(y_test, y_pred6))
print(accuracy_score(y_test, y_pred6))
print(classification_report(y_test, y_pred6))


# In[85]:


metrics.plot_roc_curve(Dtree, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred6, average= None)


# In[86]:


import joblib


# In[87]:


joblib.dump(xgb, "xgb.pkl")


# In[ ]:





# In[ ]:




