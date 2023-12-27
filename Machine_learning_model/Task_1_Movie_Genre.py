import pandas as pd
import numpy as np
import sklearn
import re
from langdetect import detect
import matplotlib as plt
import seaborn as sea
from googletrans import Translator

'''
##Loading_data
train_data= pd.read_csv(D:\\datasets\\movie_genre\\Genre Classification Dataset\\train_data.txt",
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

train_data.to_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntrain_data.txt",index=False)
test_data.to_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntest_data.txt",index=False)
print(train_data.head(2))
print(test_data.head(2))
'''
train_data = pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntrain_data.txt")
test_data = pd.read_csv("D:\\datasets\\movie_genre\\Genre Classification Dataset\\Ntest_data.txt")

train_non_english_count= train_data['Language'][train_data['Language']!='en'].value_counts().sum()
test_non_english_count= test_data['Language'][test_data['Language']!='en'].value_counts().sum()
print(f"Total train data movies' story in non-english Language is {train_non_english_count}")
print(f"Total test data movies' story in non-english Language is {test_non_english_count}")

###Conversion of other language to english language


def translate_to_english(story):
    try:
        translation = Translator().translate(story, dest='en')
        return translation.story
    except:
        return story
train_mask= train_data['Language'] != 'en'
# # train_data.loc[~train_data['Language'].isin (['en']),'story']= train_data.loc[~train_data['Language'].isin(['en']),'story'].apply(translate_to_english)
train_data.loc[train_mask,'story']= train_data.loc[train_mask,'story'].apply(translate_to_english)
def language_detection(txt):
    try:
        language = detect(txt)
        return language
    except:
        return None

train_data['NLanguage']=train_data['story'].apply(language_detection)
train_non_english_count_after_conversion= train_data['NLanguage'][train_data['NLanguage']!='en'].value_counts().sum()
print(f"Total train data movies' story after conversion in non-english Language is {train_non_english_count_after_conversion}")
#



# train_non_english_count_after_conversion= train_data['Language'][train_data['Language']!='en'].value_counts().sum()
# print(f"Total train data movies' story after conversion in non-english Language is {train_non_english_count_after_conversion}")

