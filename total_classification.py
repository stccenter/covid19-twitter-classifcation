
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
import re
import pickle

STOPWORDS = set(stopwords.words("english"))
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 2000000
EMBEDDING_DIM = 300

#path = '/Users/manzhu/Desktop/PycharmTensorflow/DisasterTweets/'
#EMBEDDING_FILE = path+'GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'


df_inform = pd.read_csv('total_labeled_utf8.csv',sep=',')
data = shuffle(df_inform)
X_inform = data.tweet_text
y_inform = data.category_id
r1 = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

for text in X_inform:
    text = re.sub(r1, "",text)
    text = filter(lambda word: word not in STOPWORDS, text)
category = to_categorical(y_inform)

tokenizer = Tokenizer(num_words=2000000)
tokenizer.fit_on_texts(X_inform)

class_sequences = tokenizer.texts_to_sequences(X_inform)
class_data = pad_sequences(class_sequences, maxlen=MAX_SEQUENCE_LENGTH)

VALIDATION_SPLIT = 0.2

nb_validation_samples = int(VALIDATION_SPLIT * class_data.shape[0])
print(category.shape, class_data.shape, nb_validation_samples)
x_train = class_data[:-nb_validation_samples]
y_train = category[:-nb_validation_samples]
x_val = class_data[-nb_validation_samples:]
y_val = category[-nb_validation_samples:]



#######################################################

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

from keras.layers import Embedding
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                            embedding_matrix.shape[1], # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Bidirectional
from keras.optimizers import RMSprop
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

########################################

print("Experiment 1")

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6), metrics=['categorical_accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=70)
#score = model.evaluate(x_val, y_val, verbose=0)
#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
# np.set_printoptions(precision=4)
# print('accuracy: ', accuracy_score(Y_val, Y_pred))
# precision, recall, fscore, support = score(Y_val, Y_pred)
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))
print(metrics.classification_report(Y_val, Y_pred))
#########

model_json = model.to_json()
with open("model_save/summary_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_save/summary_model.h5")
with open('model_save/summary_model.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved model to disk")


##########################################
''' ### EXP 2:

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
#x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
#x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
#score = model.evaluate(x_val, y_val, verbose=0)

#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
print(metrics.classification_report(Y_val, Y_pred))

'''


##########################################
''' Exp 3:

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(5,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
#score = model.evaluate(x_val, y_val, verbose=0)

#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
print(metrics.classification_report(Y_val, Y_pred))
#########
'''

#########################################
# EXP 4:
'''
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(5,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
print(metrics.classification_report(Y_val, Y_pred))
#########

'''




#############################################################
'''
##Logistic regression
##loss: 0.9084 - acc: 0.7368 - val_loss: 1.1253 - val_acc: 0.5567
model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(5, input_dim=x_train.shape[1], activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
print(metrics.classification_report(Y_val, Y_pred))
#########
'''

#############################################################

'''
##SVM

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

vectorizer = TfidfVectorizer(min_df=5,max_df = 0.75,sublinear_tf=True,use_idf=True)
classifier = LinearSVC() #svm.SVC(kernel='linear') #
# Train:
Xs = vectorizer.fit_transform(X_inform)#all_texts
classifier.fit(Xs[:-nb_validation_samples], y_inform[:-nb_validation_samples])
# Predict:
y_pred = classifier.predict(Xs[-nb_validation_samples:])

#score = cross_val_predict(classifier, Xs, y_inform[-nb_validation_samples:], cv=10) #, n_jobs=-1,Xs, category, cross_val_score

print(metrics.accuracy_score(y_inform[-nb_validation_samples:], y_pred))
print(metrics.classification_report(y_inform[-nb_validation_samples:],y_pred))
print(metrics.confusion_matrix(y_inform[-nb_validation_samples:],y_pred))
'''