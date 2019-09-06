from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Add
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from csv import DictReader
import re   
import hashlib
import score as scorer

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

        bodies = "competition_test_bodies.csv"
        stances = "competition_test_stances.csv"

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

batch_size = 1024 
epochs = 50  
latent_dim = 256 
num_samples = 30000 
max_length = 200
rejectLevel = 1
bannedWords = ['of', 'for', 'the', 'to', 'in', 'at', 'by', 'a', 'and', 'or', 'that', 'this', 'for', 'from','what', 'their', 'which', 'into', 'them', 'be', 'as', 'also', 'some', 'up', 'down', 'go', 'have', 'it', 'I', 'on', 'with']

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
            target_stance.append(stance)

            for word in headline_text:
                if word not in words:
                    words[word] = 1
                else:
                    words[word] = words[word] + 1
            for word in article_text:
                if word not in words:
                    words[word] = 1
                else:
                    words[word] = words[word] + 1
        else:
            bannedID.add(BodyID)

for word in list(words.keys()):
    if(words[word] <= rejectLevel):
        del words[word]
words['PAD'] = 0
words['UNK'] = 0
target_stance = np.asarray(target_stance)

words = set(words.keys())

num_tokens = len(words)

max_article_seq_length = max([len(txt) for txt in article_texts])
max_headline_seq_length = max([len(txt) for txt in headline_texts])

token_index = {}
for word in enumerate(words):
    m = hashlib.md5()
    b = word[1].encode('utf-8')
    m.update(b)
    x = int(m.hexdigest(), 16)
    token_index[word[1]] = x >> 64

words = None

print("")
print("Articles used: " + str(len(article_texts)))
print("Longest article: " + str(max_article_seq_length))
print("Dictionary size: " + str(num_tokens))
print("")

article_input_data = np.zeros((len(article_texts), max_article_seq_length), dtype='float64')
headline_input_data = np.zeros((len(headline_texts), max_headline_seq_length), dtype='float64')

for i, (article_text,) in enumerate(zip(article_texts)):
    for t, word in enumerate(article_text):
        if (t < (max_article_seq_length - len(article_text))):
            article_input_data[i, t] = token_index['PAD']
        else:
            try:
                article_input_data[i, t] = token_index[word]
            except KeyError:
                article_input_data[i, t] = token_index['UNK']
for i, (headline_text,) in enumerate(zip(headline_texts)):
    for t, word in enumerate(headline_text):
        if (t< (max_headline_seq_length - len(headline_text))):
            headline_input_data[i, t] = token_index['PAD']
        else:
            try:
                headline_input_data[i, t] = token_index[word]
            except KeyError:
                headline_input_data[i, t] = token_index['UNK']

article_input_data = np.reshape(article_input_data, (len(article_texts), max_article_seq_length, 1))
headline_input_data = np.reshape(headline_input_data, (len(article_texts), max_headline_seq_length, 1))

article_inputs = Input(shape=(max_article_seq_length, 1))
article = LSTM(latent_dim, return_state=True, dropout = 0.2)
article_outputs, articleState_h, articleState_c = article(article_inputs)
article_states = [articleState_h, articleState_c]

headline_inputs = Input(shape=(max_headline_seq_length, 1))
headline = LSTM(latent_dim, return_sequences = True, return_state=True, dropout = 0.2)
headline_outputs, headlineState_h, headlineState_c = headline(headline_inputs, initial_state = article_states)

added = Add()([headlineState_c, articleState_c, headlineState_h, articleState_h])
x = Dense(256)(added)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(128)(x)
x = LeakyReLU()(x)
out = Dense(4, activation = 'softmax')(x)

model = Model([article_inputs, headline_inputs], out)
model.compile(optimizer='adadelta', loss='categorical_crossentropy')

model.load_weights('s2s.h5')
arrays = model.predict([article_input_data, headline_input_data])
predictions = []
for array in arrays:
    num = np.argmax(array)
    if(num == 0):
        predictions.append("unrelated")
    elif(num == 1):
        predictions.append("agree")
    elif(num == 2):
        predictions.append("disagree")
    else:
        predictions.append("discuss")

scorer.report_score(target_stance, predictions)
