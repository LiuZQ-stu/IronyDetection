import glob
import numpy as np
from xml.dom.minidom import parse
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import  accuracy_score
from nltk.tokenize import TweetTokenizer
import emoji
import matplotlib.pyplot as plt
from nrclex import NRCLex

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
#%%
## Read Dataset

#file_name = 'pan22-author-profiling-training-2022-03-29'
# xml_paths = glob.glob(file_name+'/*/*.xml')
# truth_path = glob.glob(file_name+'/*/*.txt')
file_name = 'en'
xml_paths = glob.glob(file_name+'/*.xml')
truth_path = glob.glob(file_name+'/*.txt')
truth = open(truth_path[0], 'r').readlines()
truth = [line.strip().split(':::') for line in truth]
truth_dict = {line[0]:int(line[1]=='I') for line in truth}

dataset = dict()
labels = list()

for xml in xml_paths:
    sents = []
    user = xml.split('\\')[-1].split('.')[0]
    DOMTree = parse(xml)
    collection = DOMTree.documentElement
    documents = collection.getElementsByTagName('document')
    for document in documents:
        sent = document.childNodes[0].data
        sents.append(sent)
    dataset[user] = sents
    labels.append(truth_dict[user])

X_train, X_test, y_train, y_test = train_test_split(
    list(dataset.keys()), labels, test_size=0.2, random_state=43)

X_train_tweet = dict()
X_test_tweet = dict()

for user in X_train:
    X_train_tweet[user] = dataset[user]
for user in X_test:
    X_test_tweet[user] = dataset[user]

## Tokenizing
def tokenize_author(Xtrain):
    tk = TweetTokenizer()
    dout = dict()
    for user in Xtrain.keys():
        dout[user] = []
        for tweet in Xtrain[user]:
            dout[user].append(tk.tokenize(tweet))

    return dout
tokenized_train = tokenize_author(X_train_tweet)
tokenized_test = tokenize_author(X_test_tweet)

## Extract Features
## Statistical Features
statistics_lists = ['length', 'capital', 'questionmark','exclamationmark','period','or','hashtags','links','user_mentions','emoji']
def generate_features(tweets_tokenized):
    dout = dict()
    
    for user in tweets_tokenized.keys():
        length = []
        capital = []
        questionmark = []
        exclamationmark = []
        period = []
        or_counts = []
        hashtags = []
        links = []
        user_mentions = []
        emoji_counts = []

        for tweet in tweets_tokenized[user]:
            le = len(tweet)
            ca = sum([sum([c.isupper() for c in el]) for el in tweet])/le
            emojis = len(emoji.emoji_lis(emoji.emojize(''.join(tweet))))
            emoji_counts.append(emojis)
            qu = 0
            ex = 0
            pe = 0
            or_c = 0
            ha = 0
            us = 0
            li = 0
            T = 0
            for el in tweet:
                if el == '?':
                    qu += 1
                if el == '!':
                    ex += 1
                if el == '.':
                    pe += 1
                if el.lower() == 'or':
                    or_c += 1
                if el == '#HASHTAG':
                    ha += 1
                if el == '#USER':
                    us += 1
                if el == '#URL':
                    li += 1
                if el.lower() == 'trump':
                    T += 1
            length.append(le)
            capital.append(ca)
            questionmark.append(qu)
            exclamationmark.append(ex)
            period.append(pe)
            or_counts.append(or_c)
            hashtags.append(ha)
            links.append(li)
            user_mentions.append(us)
            #T_mentions.append(T)
                
        dout[user] = [np.mean(length),
                      np.mean(capital),
                      np.mean(questionmark),
                      np.mean(exclamationmark),
                      np.mean(period),
                      np.mean(or_c),
                      np.mean(hashtags),
                      np.mean(links),
                      np.mean(user_mentions),
                      np.mean(emoji_counts),
                      #np.mean(T_mentions)
                     ]
        
    return dout

## Emotion Features
def generate_emotion_dict(users):
    dout = dict()
    for user,tweets in users.items():
        text = ' '.join(tweets)
        text_object = NRCLex(text)
        dout[user] = list(text_object.affect_frequencies.values())
        
    return dout
        
def compose_features(statistics, emotion):
    new_dict = dict()
    for user in emotion.keys():
        new_dict[user] = statistics[user] + emotion[user]
        
    return new_dict   

text = ' '.join(dataset[X_train[0]])
text_object = NRCLex(text)
emotion_lists = list(text_object.affect_frequencies.keys())
emotion_lists

feature_lists = statistics_lists + emotion_lists
X_train_emotion = generate_emotion_dict(X_train_tweet)
X_test_emotion = generate_emotion_dict(X_test_tweet)

X_train_features = generate_features(tokenized_train)
X_test_features = generate_features(tokenized_test)
X_train_concated = compose_features(X_train_features,X_train_emotion)
X_test_concated = compose_features(X_test_features,X_test_emotion)

## Classifier

def multi_classifiers(Xtrain, Xtest, ytrain, ytest):
    # Create different classifiers
    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Decision Tree": DecisionTreeClassifier(max_depth=50),
        "Random Forest": RandomForestClassifier(max_depth=50, n_estimators=100, max_features=1),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "KMeans": KMeans(n_clusters=2),
        "Linear SVC": SVC(kernel="linear", C=1)
    }
    
    for name, classifier in classifiers.items():
        print('Classifier: ', name)
        classifier.fit(Xtrain, ytrain)
        ytrain_pred = classifier.predict(Xtrain)
        scores_train = accuracy_score(ytrain, ytrain_pred)
        print('Training Accuracy score: ', scores_train)
        ytest_pred = classifier.predict(Xtest)
        scores_test = accuracy_score(ytest, ytest_pred)
        print('Test Accuracy score: ', scores_test)

    return


## Statistical Features
multi_classifiers(list(X_train_features.values()), list(X_test_features.values()), y_train, y_test)

## Emotion Features
multi_classifiers(list(X_train_emotion.values()), list(X_test_emotion.values()), y_train, y_test)
## Features
multi_classifiers(list(X_train_concated.values()), list(X_test_concated.values()), y_train, y_test)
