import json
import os
import pickle
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


# reset initial state
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])  # input layer
# two hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# neurons represent each of our classes
net = tflearn.fully_connected(
    net, len(output[0]), activation="softmax")  # output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)

if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# convert bag of words to a numpy array that has word frequencies
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)  # list of tokenized words
    s_words = [stemmer.stem(word.lower())
               for word in s_words]  # stem words to root words

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Comece a falar com o bot")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        print(results[0][results_index])

        if results[0][results_index] > 0.95:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("Não entendi")


chat()