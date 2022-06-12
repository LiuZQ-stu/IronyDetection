import os
from xml.dom import minidom

import emoji
import ktrain
import numpy as np
from ktrain import text
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

authors = [el.split('.')[0] for el in os.listdir('pan22-author-profiling-training-2022-03-29\en')]
labels = np.loadtxt('pan22-author-profiling-training-2022-03-29/en/truth.txt', delimiter=':::', dtype=np.str)

train, test = train_test_split(authors, test_size=0.2, random_state=7)


def parse(xml):
    tweets = []
    doc = minidom.parse(xml)
    doclist = doc.getElementsByTagName('document')
    for i in range(len(doclist)):
        tweet = doclist[i].firstChild.nodeValue
        tweets.append(tweet.rstrip('\n'))

    return np.array(tweets)


def binarize_label(label):
    if label == 'NI':
        return 0
    elif label == 'I':
        return 1


def compose_dataset(labels, authors):
    dataset = {}
    dataset_for_tokenizing = {}
    labels_out = []
    for label in labels:
        id, cls = label[0], label[1]
        if id in authors:
            dataset[id] = ''
            dataset_for_tokenizing[id] = []
            tweets = parse(os.path.join('pan22-author-profiling-training-2022-03-29/en/', id + '.xml'))
            tweets = [emoji.demojize(tweet.replace('/n', '')) for tweet in tweets]
            for tweet in tweets:
                dataset[id] += tweet
                dataset_for_tokenizing[id].append(tweet)
            labels_out.append(binarize_label(cls))
    return dataset, dataset_for_tokenizing, labels_out


def tokenize_tweets(ds):
    tk = TweetTokenizer()
    dout = {}
    for author in ds.keys():
        dout[author] = []
        for tweet in ds[author]:
            dout[author].append(tk.tokenize(tweet))
    return dout


train_ds, train_tokens, labels_train = compose_dataset(labels, train)
test_ds, test_tokens, labels_test = compose_dataset(labels, test)

tokenized_train = tokenize_tweets(train_tokens)
tokenized_test = tokenize_tweets(test_tokens)

MODEL_NAME = 'bert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=30, classes=[1, 0])
trn = t.preprocess_train(list(train_ds.values()), labels_train)
tst = t.preprocess_test(list(test_ds.values()), labels_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=tst, batch_size=16)
learner.fit_onecycle(0.00001, 5)
print(learner.validate(class_names=t.get_classes()))

test = t.preprocess_test(list(test_ds.values()), labels_test)

val_pred = learner.predict(test)
print(classification_report(labels_test, np.argmax(val_pred, axis=1)))
