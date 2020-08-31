#!/usr/bin/env python
# coding: utf-8

# In[68]:


#importing necessary modules

#pandas is used for data manipulation
import pandas as pd

#importing modules from scikitlearn for machine learning
from sklearn.model_selection import train_test_split



#loading csv data file

data = pd.read_csv("forestfires.csv")


#plotting line graph of index number vs area size of fires

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


plt.plot(data.index, data.area)
plt.xlim(0, 1000)
plt.title('Range of Fire Areas')
plt.xlabel('Index Number')
plt.ylabel("Size(Hectares)")

#creating a column where the areas are give 
#a number from (0,1,2,3,4,5) depending on their size

level = []

for i in range(len(data["area"])):
    
    if data.iloc[i]["area"] < .101171:
        
        level.append(0)
    
    elif data.iloc[i]["area"] < 4.04686:
        
        level.append(1)
        
    elif data.iloc[i]["area"] < 40.4686:
        
        level.append(2)
    
    elif data.iloc[i]["area"] < 121.406:
        
        level.append(3)
    
    elif data.iloc[i]["area"] < 404.686:
        
        level.append(4)
 
    else:
        
        level.append(5)


#adding column into data frame         

data["level"] = level

data.head()


# In[57]:


#importing KNN Classifier from sklearn

from sklearn.neighbors import KNeighborsClassifier


#setting value of knn constant 

knn = KNeighborsClassifier(n_neighbors=1)

# creating X value or independent variables and y value dependent or predicted value

feature_columns = ["temp","rain","wind", "RH","DC", "ISI","DMC","FFMC"]

X = data[feature_columns]

y = data["level"]

print(knn)

knn.fit(X,y)

#generating test data using random numbers

from random import randint

temp = []
rain = []
DC = []
FFMC = []
wind = []
ISI = []
DMC = []
RH = []

for i in range(576):
    
    #ranges are established based on the maximum and minimu values of each column
    
    temp.append(randint(3,35))
    rain.append(randint(0,7))
    DC.append(randint(8,861))
    FFMC.append(randint(19, 87))
    wind.append(randint(0,10))
    ISI.append(randint(0,67))
    DMC.append(randint(1,294))
    RH.append(randint(15,100))
    
# creating test_data dataframe
d = {'temp':temp,'rain':rain,"DC":DC,"FFMC":FFMC,"wind":wind,"ISI":ISI,"DMC":DMC,"RH":RH}

test_data = pd.DataFrame(d)

test_data.head()

#Predicting Fire Level Classifictions based on Test Data

X_new = test_data


c = knn.predict(X_new)

c = c.astype(object)


# Converting the numbered ranks to lettered ones for more pleasing presentation
for i in range(len(c)):
    if c[i] == 0:
        c[i] = "Class A"
    elif c[i] == 1:
        c[i] = "Class B"
    elif c[i] == 2:
        c[i] = "Class C"
    elif c[i] == 3:
        c[i] = "Class D"
    elif c[i] == 4:
        c[i] = "Class E"
    elif c[i] == 5:
        c[i] = "Class F"
      
        
print("Testing Data")
test_data.head()       

print("Predicted Level of Fire")
print(c)



# In[20]:


#Test Data

from random import randint

temp = []
rain = []
DC = []
FFMC = []
wind = []
ISI = []
DMC = []
RH = []

for i in range(576):
    temp.append(randint(3,35))
    rain.append(randint(0,7))
    DC.append(randint(8,861))
    FFMC.append(randint(19, 87))
    wind.append(randint(0,10))
    ISI.append(randint(0,67))
    DMC.append(randint(1,294))
    RH.append(randint(15,100))
    
d = {'temp':temp,'rain':rain,"DC":DC,"FFMC":FFMC,"wind":wind,"ISI":ISI,"DMC":DMC,"RH":RH}

test_data = pd.DataFrame(d)

test_data.head


# In[37]:


#imporitng Grid search to determine best constant value for knn

from sklearn.model_selection import GridSearchCV

k_range = range(1,31)

param = dict()

param_grid = dict(n_neighbors = k_range)

print(param_grid)

## initiate grid

## data will split up into 10 sections and cross validated against eachother
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'accuracy')

grid.fit(X,y)


# In[64]:


#putting results of gridsearchcv in pandas dataframe 

pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]


# In[41]:


#graph that shows the accuracy of respective k values for knn

plt.plot(k_range, grid_mean_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")


# In[49]:


#outputs best k value for knn

print(grid.best_score_)
print(grid.best_params_)


# In[70]:



knn2 = KNeighborsClassifier(n_neighbors=14)

knn2.fit(X,y)

predictions = knn.predict(test_data)


print(predictions)


# In[46]:


#Here we compare KNN and Logistic Regression models with crossvalidation 
#by splitting the data into 10 sections and testing against eachother

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# In[47]:


#Logistic Regressions shows more accuracy

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())


# In[66]:


#use logistic regression model to predict wildfire classifications 

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X, y)


y_pred = logreg.predict(test_data)

print(y_pred)

