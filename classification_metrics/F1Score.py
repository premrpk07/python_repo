#!/usr/bin/env python
# coding: utf-8

# ####  Script Name: F1 Score Calculation
# ####  Purpose: To calcualte F1 score,Precision and Recall for given file input for Thursday
# ####  Input: test.psv
# ####  Output: 
# ####  Version: 0.1
# ####  Author: Pem Kumar Rajendran
# ####  Block1: Library used Pandas,Numpy,matplot,sklearn(for Calculating F1 score)
# ####  Block2: Library used Pandas,Numpy,matplot and implemented one logic for calculatinf F1 score without external library

# ##  Block1: Library used Pandas,Numpy,matplot,sklearn(for Calculating F1 score)

# In[3]:


import datetime as dt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import matplotlib.pyplot as plt

def day_val(rowdata):
    
## Function to return Day Names
## Input Dataframe rows
## Output: Day names
## Eg: 2020-09-05 as Thursday

    val=dt.datetime.strptime(rowdata['dates'], '%Y-%m-%d').weekday()
    #val=rowdata['dates']
    return val


# Read file #
test_date_file='test.psv'
input_dataframe=pd.read_csv(test_date_file, sep='|',  parse_dates=True,skiprows=1,header=0)

print("Top 5 Lines of Input: \n")
print(input_dataframe.head(5))
print("\n")
## Create new column for Day names by apply funtion and filter rows for Thursday
input_dataframe['day_val'] = input_dataframe.apply(day_val, axis=1)

sample_numbers = 20

print("Initial Data Exploration based on y: \n")
print("\n")

## Label 0 Initial Data Analysis
class_train_frame_1 = input_dataframe.query('y == 0')[['y','day_val']]
plt.clf()
class_train_frame_1.groupby('day_val').count().plot(kind='bar')
plt.show()
print("\n")
## Label 0 Initial Data Analysis
class_train_frame_2 = input_dataframe.query('y == 1')[['y','day_val']]
plt.clf()
class_train_frame_2.groupby('day_val').count().plot(kind='bar',color = 'r')
plt.show()
print("\n")
del class_train_frame_2,class_train_frame_1

epoch=[0,1,2,3,4,5,6]
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
Epoch=[]
fold_n=[]
week_dict={0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
for i in epoch:
    day_val=i
    input_dataframe_t = input_dataframe.query('day_val == @day_val')
    y_actual=input_dataframe_t['y'].to_numpy()
    y_predict=input_dataframe_t['yhat'].to_numpy()   
    #cm=build_confusion_matrix(y_actual,y_predict)
    rf_accuracy_score = accuracy_score(y_actual, y_predict)
    rf_precision_score = precision_score(y_actual, y_predict, average='macro')
    rf_f1_score = f1_score(y_actual, y_predict, average='macro')
    rf_recall_score = recall_score(y_actual, y_predict, average='macro')
    accuracy_scores.append(rf_accuracy_score*100)
    f1_scores.append(rf_f1_score*100)
    precision_scores.append(rf_precision_score*100)
    recall_scores.append(rf_recall_score*100)
    Epoch.append(week_dict.get(i))
    fold_n.append(i)
    
report_frame = pd.concat([pd.Series(Epoch,name='Epoch'),pd.Series(fold_n,name='fold_n'),pd.Series(accuracy_scores,name='accuracy_score'),pd.Series(f1_scores,name='f1_score'),pd.Series(precision_scores,name='precision_score'),pd.Series(recall_scores,name='recall_score')], axis=1)

print("Metrics for All Days Epoch: \n")
print("\n")
print(report_frame  )




print ('Metrics: ')
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlim(0, 7)
host.set_ylim(0, 100)
par1.set_ylim(0, 100)
par2.set_ylim(1, 100)

host.set_xlabel("Epoch")
host.set_ylabel("accuracy_score")
par1.set_ylabel("precision_score")
par2.set_ylabel("f1_score")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p3, = host.plot(report_frame['Epoch'].str.slice(stop=3), report_frame['accuracy_score'], color=color1,label="accuracy_score")
p2, = par1.plot(report_frame['Epoch'].str.slice(stop=3), report_frame['precision_score'], color=color2, label="precision_score")
p1, = par2.plot(report_frame['Epoch'].str.slice(stop=3), report_frame['f1_score'], color=color3, label="f1_score")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))      
# no x-ticks                 
par2.xaxis.set_ticks(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

print("\n")
print("Metrics in Line Plot \n")
print("\n")
plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')
plt.show()
