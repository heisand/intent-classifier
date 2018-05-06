import json
from pprint import pprint
import numpy as np
import keras
import keras.utils
from keras import utils as np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import sklearn
from sklearn.cross_validation import train_test_split

texts = []
labels_index = {}
labels = []
labels_id = []

data = json.load(open('bot_samples.json'))

label_count = 0

for dialogue in data["dialogues"]:
        if ("nb" in dialogue["samples"].keys()):
                for question in dialogue["samples"]["nb"]:
                        if ("nb" in dialogue["replies"].keys()):
                                for reply in dialogue["replies"]["nb"]:
                                        texts.append(question)
                                        label_id = len(labels_index)
                                        labels_index[reply] = label_id
                                        labels.append(label_id)
        if ("en" in dialogue["samples"].keys()):
                for question in dialogue["samples"]["en"]:
                        if ("en" in dialogue["replies"].keys()):
                                for reply in dialogue["replies"]["en"]:
                                        texts.append(question)
                                        label_id = len(labels_index)
                                        labels_index[reply] = label_id
                                        labels.append(label_id)
                             

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences)

labels = np_utils.to_categorical(np.asarray(labels))
#print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

#print (data)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train, x_validate, x_test = np.split(data, [int(.8*len(data)), int(.9*len(data))])
y_train, y_validate, y_test = np.split(data, [int(.8*len(data)), int(.9*len(data))])

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

#print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(29, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)


tokenizer.fit_on_texts("Nei nei nei")
prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences("Nei nei nei")))
#print(prediction)
