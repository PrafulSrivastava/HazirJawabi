import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import pickle
import random
import json

with open("intents.json") as file:
    data = json.load(file)
try:

    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
# print(data)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            # print(wrds)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    print(words)
    print(docs_x)
    print(docs_y)
    print(labels)

    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!"]]

    print(words)
    words = sorted(list(set(words)))
    print(words)
    labels = sorted(labels)
    print(labels)

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
        print(bag)
        output_row = out_empty[:]
        print(output_row)
        output_row[labels.index(docs_y[x])] = 1
        print(output_row)
        training.append(bag)

        output.append(output_row)
    print(training)
    training = numpy.array(training)
    print(training)
    print(output)
    output = numpy.array(output)
    print(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
nrt = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w) for w in s_words]
    for se in s_words:
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Enter query")
    while True:
        ip = input("You: ")
        if ip.lower() == "quit":
            break
        bow = bag_of_words(ip, words)
        print(bow)
        result = model.predict([bow])[0]
        res_ind = numpy.argmax(result)
        tag = labels[res_ind]
        if result[res_ind] > 0.5:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    resp = tg["responses"]

            print(random.choice(resp))
        else:
            print("Didn't get you dude! Try again maybe")


chat()
