from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adadelta
import numpy as np
from csv import DictReader
import re   
import hashlib
from gensim.models import KeyedVectors, Word2Vec
import keras.backend as K
from itertools import product
def classify(string):
    if (string == "unrelated"):
        return [1,0,0,0]
    if (string == "agree"):
        return [0,1,0,0]
    if (string == "disagree"):
        return [0,0,1,0]
    if (string == "discuss"):
        return [0,0,0,1]

class DataSet():
    def __init__(self, name="train", path="fnc-1"):
        self.path = path

        bodies = "train_bodies.csv"
        stances = "train_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']
 
    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

batch_size = 256 
epochs = 500  
latent_dim = 128 
num_samples = 30000 
max_length = 200
rejectLevel = 1
bannedWords = ['of', 'for', 'the', 'to', 'in', 'at', 'by', 'a', 'and', 'or', 'that', 'this', 'for', 'from','what', 'their', 'which', 'into', 'them', 'be', 'as', 'also', 'some', 'up', 'down', 'go', 'have', 'it', 'I', 'on', 'with']

fileName = '/home/david/Documents/NeuralNets/FakeNews/utils/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(fileName, binary = True)
dataset = DataSet()

article_texts = []
headline_texts = []
target_stance = []
words = dict()
bannedID = set()

for i in range(len(dataset.stances)): 
    headline = dataset.stances[i]['Headline']
    BodyID = dataset.stances[i]['Body ID']
    stance = dataset.stances[i]['Stance']
    if(BodyID not in bannedID): 
        article_text = dataset.articles[BodyID]
        article_text = re.sub(r'[^\w\s]','',article_text)
        headline = re.sub(r'[^\w\s]','',headline)
        for word in bannedWords:
            article_text = re.sub(r'(\s{word}\s)'.format(word = word), ' ', article_text)
            headline = re.sub(r'(\s{word}\s)'.format(word = word), ' ', headline)

        headline_text = headline.split()
        article_text = article_text.split()

        if(len(article_text) <= max_length):
            headline_texts.append(headline_text)
            article_texts.append(article_text)
            target_stance.append(classify(stance))

            for word in headline_text:
                word = word.lower()
                if word not in words:
                    words[word] = 1
                else:
                    words[word] = words[word] + 1
            for word in article_text:
                word = word.lower()
                if word not in words:
                    words[word] = 1
                else:
                    words[word] = words[word] + 1
        else:
            bannedID.add(BodyID)

target_stance = np.asarray(target_stance)

words = set(words.keys())
dataset = None
num_tokens = len(words)

max_article_seq_length = max([len(txt) for txt in article_texts])
max_headline_seq_length = max([len(txt) for txt in headline_texts])

print("")
print("Articles used: " + str(len(article_texts)))
print("Longest article: " + str(max_article_seq_length))
print("Dictionary size: " + str(num_tokens))
print("")

article_input_data = np.zeros((len(article_texts), max_article_seq_length, 300), dtype='float32')
headline_input_data = np.zeros((len(headline_texts), max_headline_seq_length, 300), dtype='float32')
pad = np.zeros(300, dtype = 'float32')
unk = np.ones(300, dtype = 'float32')

for i, (article_text,) in enumerate(zip(article_texts)):
    for t, word in enumerate(article_text):
        if (t < (max_article_seq_length - len(article_text))):
            article_input_data[i,t] = pad            
        else:
            try:
                vector = model[word]
                article_input_data[i, t] = vector
            except KeyError:
                article_input_data[i, t] = unk
for i, (headline_text,) in enumerate(zip(headline_texts)):
    for t, word in enumerate(headline_text):
        if (t< (max_headline_seq_length - len(headline_text))):
            headline_input_data[i, t] = pad
        else:
            try:
                vector = model[word]
                headline_input_data[i, t] = vector
            except KeyError:
                headline_input_data[i, t] = unk

article_input_data = np.reshape(article_input_data, (len(article_texts), max_article_seq_length, 300))
headline_input_data = np.reshape(headline_input_data, (len(article_texts), max_headline_seq_length, 300))

article_inputs = Input(shape=(None, 300))
article = LSTM(latent_dim, return_state=True, dropout = 0.2)
article_outputs, articleState_h, articleState_c = article(article_inputs)
article_states = [articleState_h, articleState_c]

headline_inputs = Input(shape=(None, 300))
headline = LSTM(latent_dim, return_sequences = True, return_state=True, dropout = 0.2)
headline_outputs, headlineState_h, headlineState_c = headline(headline_inputs, initial_state = article_states)

added = Add()([headlineState_c, articleState_c, headlineState_h, articleState_h])
x = Dense(128)(added)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(64)(x)
x = LeakyReLU()(x)
out = Dense(4, activation = 'softmax')(x)

model = Model([article_inputs, headline_inputs], out)
adadelta = Adadelta(lr = 2)
model.compile(optimizer=adadelta, loss='categorical_crossentropy')

class_weight = {0: 1., 1: 7., 2: 7., 3: 5.}

print(model.summary())


model.fit([article_input_data, headline_input_data], target_stance, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save('s2s.h5')


