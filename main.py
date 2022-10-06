import  numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#loading the data from csv file
raw_mail_data=pd.read_csv('D:\pyhton\mail_data.csv')

print(raw_mail_data)
#replacing the null value with a null string
raw_mail=raw_mail_data.where(pd.notnull(raw_mail_data),'')

raw_mail.shape


#rename the categrories ham:0 and spam :1
raw_mail.loc[raw_mail['Category'] == 'spam', 'Category',] = 0
raw_mail.loc[raw_mail['Category'] == 'ham', 'Category',] = 1

print('renaming the categories spam : 0 and ham : 1 ')

#seperating the columns

messages= raw_mail['Message']
categories=raw_mail['Category']

print(categories)

#spliting thre data into training set and test set

messages_train,messages_test,categories_train,categories_test=train_test_split(messages,categories,test_size=0.2,random_state=3)
print(messages.shape)
print(messages_train.shape)
print(messages_test.shape)