import  numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#loading the data from csv file
raw_mail_data=pd.read_csv('D:\pyhton\mail_data.csv')

print(raw_mail_data)
