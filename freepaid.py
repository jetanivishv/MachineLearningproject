import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("DATASET/udemy_courses.csv")

def converthours(val):
    values=val.split()
#     print(values)
    if values[0]=='0':
        return 0
    elif values[1]=='hours' or values[1]=='hour':
        return float(values[0])
    elif values[1]=='mins':
        return float(values[0])/60

df['content_duration']=df['content_duration'].apply(converthours)
df.isna().any()
df[df['content_duration'].isna()]=0
df[df['published_timestamp']==0]='2012'

def getyear(val):
    return val[0:4]
df['year']=df['published_timestamp'].apply(getyear)
df.head(1)

df.drop(['course_id','course_title','url','published_timestamp','clean_course_title','price'],axis=1,inplace=True)
df.head(1)

def conversion(val):
    if val=='TRUE':
        return 1
    else:
        return 0

df['is_paid']=df['is_paid'].apply(conversion)

def lev(val): 
  
    if val == 'All Levels': 
        return 0
    elif val == 'Intermediate Level': 
        return 1
    elif val == 'Beginner Level': 
        return 2
    else:
        return 3
    
df['level'] = df['level'].apply(lev)

def sub(val): 
  
    if val == 'Business Finance': 
        return 0
    elif val == 'Graphic Design': 
        return 1
    elif val == 'Musical Instruments': 
        return 2
    else:
        return 3
    
df['subject'] = df['subject'].apply(sub)

def year(val): 
#     print(val==2017)
#     print(type(val),val,type('2017'))
    if val == '2017': 
        return 6
    elif val == '2016': 
        return 5
    elif val == '2015': 
        return 4
    elif val == '2014': 
        return 3
    elif val == '2013': 
        return 2
    elif val == '2012': 
        return 1
    else:
        return 0
    
df['year'] = df['year'].apply(year)

x=df.drop(['is_paid'],axis=1)
y=df['is_paid']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

kmodel = KNeighborsClassifier(n_neighbors=3)
kmodel.fit(x_train,y_train)
kpredictions = kmodel.predict(x_test)
print(kpredictions,y_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,kpredictions))

kscore = round((kmodel.score(x_test, y_test)*100),2)
print ("\nModel Score:",kscore,"%")

import pickle
pickle_out = open("paid_free.pkl", "wb")
pickle.dump(kmodel, pickle_out)
pickle_out.close()


