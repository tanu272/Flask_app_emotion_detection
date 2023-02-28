mport tensorflow
import numpy as np
import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import speech_recognition as sr

lb = LabelEncoder()
stopwords = set(nltk.corpus.stopwords.words('english'))
vocab_size = 10000
len_sentence = 150


model = tensorflow.keras.models.load_model('./model.h5')
model.summary()

def text_prepare(data, column):
    print(data.shape)
    stemmer = PorterStemmer()
    corpus = []
    
    for text in data[column]:
        text = re.sub("[^a-zA-Z]", " ", text)
        
        text = text.lower()
        text = text.split()
        
        text = [stemmer.stem(word) for word in text if word not in stopwords]
        text = " ".join(text)
        
        corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=vocab_size) for word in corpus]
    embeddec_doc = pad_sequences(sequences=one_hot_word,
                              maxlen=len_sentence,
                              padding="pre")
    print(data.shape)
    return embeddec_doc

# Evaluate the model
test_data = pd.read_csv("./test.txt", header=None, sep=";", names=["Comment","Emotion"], encoding="utf-8")
test_data["Emotion"] = lb.fit_transform(test_data["Emotion"])
x_test=text_prepare(test_data, "Comment")

y_test=test_data["Emotion"]
y_test = np.array(y_test)
enc = OneHotEncoder()
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

results = model.evaluate(x_test, y_test, verbose=2)
print("test loss, test acc:", results)


def text_prepare_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    print(text)

    one_hot_word = one_hot(input_text=text, n=vocab_size)
    print([one_hot_word])
    embeddec_doc = pad_sequences(sequences=[one_hot_word],
                              maxlen=len_sentence,
                              padding="pre")
    # # print(text.shape)
    return embeddec_doc


## recognize audio
""" r = sr.Recognizer()
m = sr.Microphone()

try:
    with m as source: 
        r.adjust_for_ambient_noise(source)
    #print("Set minimum energy threshold to {}".format(r.energy_threshold))
    print("Say something!")
    with m as source: 
        audio = r.listen(source)
    print("Got it! Now to recognize it...")
    try:
        value = r.recognize_google(audio, language='en-GB')
        print(value)
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    pass """

##use audio statement to check speaker's feelings
value="im feeling good"
mod_text = text_prepare_text(value)
mod_text.shape


a=(model.predict(mod_text))
b=(model.predict(mod_text).argmax())
#print(a)
print(b)


y=['anger','fear','joy','love','sadness','surprise']
lb.fit(y)

final_pred=lb.inverse_transform(b.ravel())
print(final_pred) 
