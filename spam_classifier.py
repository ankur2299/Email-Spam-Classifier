import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

messages = pd.read_csv('./SMSSpamCollection',sep='\t',names=['label','message'])


ps = PorterStemmer()
corpus = []
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    
    for word in review:
        if word not in stopwords.words('english'):
            review = ps.stem(word)
            
    review = ' '.join(review)
    corpus.append(review)
    
    
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.20, random_state=0)


spam_detect_model = MultinomialNB().fit(x_train,y_train)

y_pred = spam_detect_model.predict(x_test)
email = input()
confusion_m = confusion_matrix(y_test,y_pred)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:-",accuracy)
