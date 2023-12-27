import pandas as pd
import numpy as np
import sklearn
import re
from langdetect import detect
import matplotlib as plt
import seaborn as sea
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import nltk.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

##Loading_data
train_data= pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\train_data.txt",
                       sep=':::', names=['Title','Genre','story'], engine= 'python')
                       # names=["Titles,Genre,Description"])
test_data= pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\test_data.txt",
                       sep=':::', names=['Title','story'], engine= 'python')
# print(test_data.head())

##Cleaning data
"""
print(train_data.duplicated().sum())
print(train_data.isna().sum())
print(train_data.info())
print(test_data.duplicated().sum())
print(test_data.isna().sum())
print(test_data.info())
"""
# There is no any duplicated and missing data

##remove unncessary characters
def cleaning_text(txt):
    txt= re.sub(r"http\S+", '', txt) # remove URL
    txt = re.sub(r'https?://\S+', '', txt) #remove URL
    txt= re.sub(r"@\S+", '', txt) # remove twitter handle
    # txt= re.sub(r"\d+", '', txt) #remove numbers
    txt= re.sub(r"[^a-zA-Z ]", '',txt) # keep only letters and spaces (remove digits and characters)
    txt= re.sub(r"\s[\s]+", '', txt) # Delete more than one spaces
    txt= txt.lower()
    return txt

train_data['story']=train_data['story'].apply(cleaning_text)
test_data['story']=test_data['story'].apply(cleaning_text)
# print(train_data['story'].head(1))

###Check for languages
def language_detection(txt):
    try:
        language = detect(txt)
        return language
    except:
        return None

train_data['Language']=train_data['story'].apply(language_detection)
test_data['Language']=test_data['story'].apply(language_detection)
'''
train_data.to_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntrain_data.txt",index=False)
test_data.to_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntest_data.txt",index=False)
print(train_data.head(2))
print(test_data.head(2))
train_data = pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntrain_data.txt")
test_data = pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntest_data.txt")
'''
train_non_english_count= train_data['Language'][train_data['Language']!='en'].value_counts().sum()
test_non_english_count= test_data['Language'][test_data['Language']!='en'].value_counts().sum()
# print(f"Total train data movies' story in non-english Language is {train_non_english_count}")
# print(f"Total test data movies' story in non-english Language is {test_non_english_count}")

##Conversion of other language to english language
def translate_to_english(story):
    try:
        translation = Translator().translate(story, dest='en')
        return translation.story
    except:
        return story
train_mask= train_data['Language'] != 'en'
test_mask= test_data['Language'] != 'en'
train_data.loc[train_mask,'story']= train_data.loc[train_mask,'story'].apply(translate_to_english)
test_data.loc[test_mask,'story']= test_data.loc[test_mask,'story'].apply(translate_to_english)

# def language_detection(txt):
#     try:
#         language = detect(txt)
#         return language
#     except:
#         return None
#
# train_data['NLanguage']=train_data['story'].apply(language_detection)
# train_non_english_count_after_conversion= train_data['NLanguage'][train_data['NLanguage']!='en'].value_counts().sum()
# print(f"Total train data movies' story after conversion in non-english Language is {train_non_english_count_after_conversion}")

## Tokenization,lemmatization and tf-Idf
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer= WordNetLemmatizer()
# print(stopwords)
def preprocess(text):
    tokens= word_tokenize(text,language="english")
    filtered_tokens= [word for word in tokens if word not in stop_words]
    lemmatized_tokens= [lemmatizer.lemmatize(word) for word in filtered_tokens]
    preprocess_story= ' '.join(lemmatized_tokens)
    return preprocess_story

train_data['story']=train_data['story'].apply(preprocess)
test_data['story']=test_data['story'].apply(preprocess)
# print(train_data['story'].head(3))

##Apply TF-IDF in preprocessed text
tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=2,max_features=2500) # take one word at time and ignore words which apperas less than 2 times
final_train_data = tfidf.fit_transform(train_data['story'])          # max feature so that model doesnot expect different fetures .
final_test_data = tfidf.fit_transform(test_data['story'])
# print(final_train_data[1])

##resamples of training data
X,Y = resample(final_train_data,train_data['Genre'])
# print(X.shape[0])
# print(Y.shape[0])
x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.3,random_state=42)


##Model Building
model=MultinomialNB(alpha=0.1)
model.fit(x_train,y_train)
y_valpred=model.predict(x_val)
y_trainpred =model.predict(x_train)
accuracy_val=accuracy_score(y_val,y_valpred)
accuracy_train=accuracy_score(y_train,y_trainpred)
print(f"The accuracy of model for validation data is {accuracy_val:.2f}")
print(f"The accuracy of model for train data is {accuracy_train:.2f}")
print(f"The Classification Report for validation is :{classification_report(y_val,y_valpred,zero_division=1.0)}")
print(f"The Classification Report for Train is :{classification_report(y_train,y_trainpred,zero_division=1.0)}")

##prediction for test data
y_testpred=model.predict(final_test_data)
test_solutions=pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\test_data_solution.txt",
                        sep=':::', names=['Title','Genre','story'], engine= 'python')
test_actual_genre= test_solutions['Genre']
accuracy_test=accuracy_score(test_actual_genre,y_testpred)
print(f"The accuracy of model for train data is {accuracy_test}")

