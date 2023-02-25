
from math import expm1
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
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

lb = LabelEncoder()
stopwords = set(nltk.corpus.stopwords.words('english'))
vocab_size = 10000
len_sentence = 150
enc = OneHotEncoder()


app = Flask(__name__)
model = keras.models.load_model("./model.h5")
#transformer = joblib.load("assets/data_transformer.joblib")

def text_prepare_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    #print(text)
    one_hot_word = one_hot(input_text=text, n=vocab_size)
    print([one_hot_word])
    embeddec_doc = pad_sequences(sequences=[one_hot_word], maxlen=len_sentence, padding="pre")
    # # print(text.shape)
    return embeddec_doc


def prepredict(value):
    mod_text = text_prepare_text(value)
    a=(model.predict(mod_text))
    b=(model.predict(mod_text).argmax())
    y=['anger','fear','joy','love','sadness','surprise']
    lb.fit(y)
    final_pred=lb.inverse_transform(b.ravel())
    return final_pred

@app.route("/", methods=["GET","POST"])
def index():
    print("[+] request received")
    # get the data from the request and put ir under the right format
    req = request.get_json(force=True)
    predicted_emo = prepredict(req)
    return jsonify({"emo": str(predicted_emo)})

