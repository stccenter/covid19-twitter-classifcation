# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import csv
import re
from string import digits  
import random
import pickle

import codecs,sys

from keras.layers import Embedding

from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, merge, GRU, Embedding, Dropout, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

from keras.models import load_model
import tensorflow as tf 
import time

start = time.time()

MAX_SEQUENCE_LENGTH = 30
STOPWORDS = set(stopwords.words("english"))
file_inform = pd.read_csv('data/covid_jul.csv',sep=',',low_memory=False)
X_inform = file_inform.tweet_text
message_id = file_inform.message_id
latitude = file_inform.latitude
longitude = file_inform.longitude
print("model_length: " + str(len(X_inform)))

r1 = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

for text in X_inform:
    text = re.sub(r1, "",text)
    text = re.sub(r'https://.+', '', text)
    text = re.sub(r'http://.+', '', text)
    text = filter(lambda word: word not in STOPWORDS, text)


tokenizer = Tokenizer()
with open('model_save/summary_model.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class_sequences = tokenizer.texts_to_sequences(X_inform)
class_data = pad_sequences(class_sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_val = class_data[0:]
#y_val = category[0:]


# load json and create model
json_file = open('model_save/summary_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_save/summary_model.h5")


#loaded_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False), metrics=['categorical_accuracy'])
#score = loaded_model.evaluate(x_val, y_val, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

all_rate=loaded_model.predict(x_val)
#all_sequence, all_final_choice = best_three_choice(all_rate)

Y_pred = np.argmax(all_rate,axis=1)
#ThemeTagDic={0:'Economic',1:'Others',2:'Political',3:'Prevention',4:'Symptoms',5:'Transmission',6:'Treatment'}
ThemeTagDic={0:'Economic',1:'Political',2:'Prevention',3:'Others',4:'Treatment',5:'Transmission',6:'Symptoms'}
final_category_list = []
Y_pred = Y_pred.tolist()

for pred_num in Y_pred:
    pred_num=ThemeTagDic[pred_num]
    final_category_list.append(pred_num)


#category_csv=pd.DataFrame({'Category':Y_pred})

#category_csv.to_csv("category_test.csv", index=False, sep=',')
#category_newform = pd.read_csv('category_test.csv',sep=',',low_memory=False)

category_english_csv=pd.DataFrame({'all_final_result_english':final_category_list})
category_english_csv.to_csv("category_test_english.csv", index=False, sep=',')

merge_id=pd.merge(X_inform,message_id,left_index=True,right_index=True,sort=False)
merge_lat=pd.merge(merge_id,latitude,left_index=True,right_index=True,sort=False)
merge_long=pd.merge(merge_lat,longitude,left_index=True,right_index=True,sort=False)
final_category_english=pd.merge(merge_long,category_english_csv,left_index=True,right_index=True,sort=False)
final_category_english.to_csv("result/Jul_result.csv", index=False, sep=',')

end = time.time()
print (str(end-start))