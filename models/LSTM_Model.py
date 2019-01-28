import time

prgStart = time.time()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

from functools import reduce 

from nltk.corpus import stopwords
import pandas as pd 
import nltk.data
import logging

# NLP Tools
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import csv

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Spell Check
import re
from collections import Counter

print("Loading files..")
file_load_start = time.time()
training_set = pd.read_csv('/media/hduser/OS_Install/Mechanical Engineering/Sem V/Data Analytics/Project/FinalTrain.csv', header = 0, delimiter = "\t", quoting = 3)
testing_set = pd.read_csv('/media/hduser/OS_Install/Mechanical Engineering/Sem V/Data Analytics/Project/FinalTest.csv', header = 0, delimiter = "\t", quoting = 3)
print("Files loaded in : ",time.time()-file_load_start,"s.\n")

#Loading punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Loading keras tokenizer
t = Tokenizer()
t.fit_on_texts(training_set["essay"])
word_index = t.word_index
vocab_size = len(t.word_index) + 1

#Preparing data for input to Word2Vec
training_essay_data = []
testing_essay_data = []

print("Training set processing..")
start = time.time()
for i in range(7, 9):
	set_in_consideration = training_set[training_set["essay_set"] == i]
	for essay in set_in_consideration["essay"]:
		training_essay_data.append(essay)
print("Processing Done in : ",time.time()-start,"s.\n")

print("Testing set processing..")
start = time.time()
for i in range(7, 9):
	set_in_consideration = testing_set[testing_set["essay_set"] == i]
	for essay in set_in_consideration["essay"]:
		testing_essay_data.append(essay)
print("Processing Done in : ",time.time()-start,"s.\n")

#Extracting Labels

train_labels = []
for i in range(7, 9):
	set_in_consideration = training_set[training_set["essay_set"] == i]
	for domain1_score in set_in_consideration["domain1_score"]:
		if(i==1):
			train_labels.append(round(domain1_score))
		elif(i==2):
			train_labels.append(round(domain1_score))
		elif(i==3):
			train_labels.append(round(domain1_score))
		elif(i==4):
			train_labels.append(round(domain1_score))
		elif(i==5):
			train_labels.append(round(domain1_score))
		elif(i==6):
			train_labels.append(round(domain1_score))
		elif(i==7):
			train_labels.append(round(domain1_score))
		elif(i==8):
			train_labels.append(round(domain1_score))			

test_labels = []
for i in range(7, 9):
	set_in_consideration = testing_set[testing_set["essay_set"] == i]
	for domain1_score in set_in_consideration["domain1_score"]:
		if(i==1):
			test_labels.append(round(domain1_score))
		elif(i==2):
			test_labels.append(round(domain1_score))
		elif(i==3):
			test_labels.append(round(domain1_score))
		elif(i==4):
			test_labels.append(round(domain1_score))
		elif(i==5):
			test_labels.append(round(domain1_score))
		elif(i==6):
			test_labels.append(round(domain1_score))
		elif(i==7):
			test_labels.append(round(domain1_score))
		elif(i==8):
			test_labels.append(round(domain1_score))

#LSTM
#Encoding the essays
encoded_train = t.texts_to_sequences(training_essay_data)
encoded_test = t.texts_to_sequences(testing_essay_data)

#Adding Padding
max_length = 500
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

#Loading the embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors' % len(embeddings_index))

#Building the embedding_matrix
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

print('Building model...')
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(61, activation='softmax'))

# tCompiling the Model
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Starting to train the Model
s = time.time()
print('Training...')
model.fit(padded_train, train_labels, epochs = 20)

#Evaluation of the LSTM Model
scores = model.evaluate(padded_test, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("LSTM Network trained in : ",time.time()-s,"s.\n")

#Testing the LSTM Model 
prediction = model.predict(padded_test)
actual_predictions = []
for i in range(0,len(prediction)):
	maxi = -1
	pos = 0
	for j in range(0, len(prediction[i])):
		if(prediction[i][j] > maxi):
			maxi = prediction[i][j]
			pos = j
	actual_predictions.append(pos)

print(actual_predictions)

def confusion_matrix(rater_a, rater_b,
		 min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a)==len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a,b in zip(rater_a,rater_b):
        conf_mat[a-min_rating][b-min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None: min_rating = reduce(min, ratings)
    if max_rating is None: max_rating = reduce(max, ratings)
    num_ratings = max_rating - min_rating + 1
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
	    hist_ratings[r-min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b,
                             min_rating = None, max_rating = None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
				     min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i]*hist_rater_b[j]
			      / num_scored_items)
            d = pow(i-j,2.0) / pow(num_ratings-1, 2.0)
            numerator += d*conf_mat[i][j] / num_scored_items
            denominator += d*expected_count / num_scored_items

    return 1.0 - numerator / denominator

qwk3 = quadratic_weighted_kappa(test_labels, actual_predictions)

print("Quadratic Weighted Kappa with LSTM Network: ", qwk3)
print("\nTime for entire execution : ",time.time()-prgStart,"s.\n")