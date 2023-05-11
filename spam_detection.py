import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

data = pd.read_csv('/content/drive/MyDrive/spam (1).csv',encoding = 'ISO-8859-1')
data.head()

data.info()

data.rename(columns = {'v1' : 'class', 'v2' : 'test'}, inplace = True)
data.head()

data.shape

data.isnull().sum()

encoder = LabelEncoder()
data['class'] = encoder.fit_transform(data['class'])
data.head()

#adding a new column
data['No char'] = data['test'].apply(len)
data.head()

data.groupby('class').count()

import string
def msg(message):
  clean_data = [char for char in message if char not in string.punctuation]
  clean_data = "".join(clean_data)
  return clean_data

data['test'] = data['test'].apply(msg)
data.head()

cv = CountVectorizer(stop_words = 'english')

x = data['test'].values
y = data['class'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)

x_train_cv = cv.fit_transform(x_train)

NBM = MultinomialNB()
NBM.fit(x_train_cv,y_train)

x_test_cv = cv.transform(x_test)
y_predict = NBM.predict(x_test_cv)
accuracy = accuracy_score(y_test, y_predict)*100
print('Acuuracy = ',accuracy)

message = input('Enter Message = ')
msg = cv.transform([message])
predict = NBM.predict(msg)
if(predict[0] == 0):
  print('!!Check spam!!')
else:
  print('**Check inbox**')
#We need to give input 
