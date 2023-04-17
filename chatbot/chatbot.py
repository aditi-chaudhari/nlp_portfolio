import random
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf

import nltk
from keras.saving.save import load_model
from nltk.stem import WordNetLemmatizer

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()
intents_file = json.loads(open('intents.json').read())

def model():
    # initialize empty lists for words in the patterns, tags, and pattern/tag tuples
    words = []
    tags = []
    patterns = []

    # create lists for the words present throughout the patterns,
    # the tags,
    # and a list of tuples which contain the words in the pattern and the tag for each pattern
    for intent in intents_file['intents']:
        for pattern in intent['patterns']:
            words_in_pattern = nltk.word_tokenize(pattern)
            words.extend(words_in_pattern)

            tag = intent['tag']
            if tag not in tags:
                tags.append(tag)

            patterns.append((words_in_pattern, tag))

    # tokenize and then lemmatize the user input. eliminate punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    words = sorted(set(words))

    # creates a list of training data tuples, where each tuple is a bag of words concatenated with a one-hot encoded tag vector
    training_data = []
    tag_indices = {tag: i for i, tag in enumerate(tags)}
    for pattern, tag in patterns:
        bag_of_words = [1 if word in pattern else 0 for word in words]
        tag_vector = [0] * len(tags)
        tag_vector[tag_indices[tag]] = 1
        training_data.append((bag_of_words, tag_vector))

    # randomly shuffle the training data and convert it to a numpy array
    random.shuffle(training_data)
    train_x = np.array([data[0] for data in training_data])
    train_y = np.array([data[1] for data in training_data])

    # build the model using deep learning
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(words),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(tags), activation='softmax')
    ])

    # compile and fit the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=200, batch_size=25, verbose=1)

    # save the model and word/tag lists
    model.save('chatbot_model.h5')
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(tags, open('tags.pkl', 'wb'))


def predict_class(sentence):
    # load words, classes, and the model
    words = pickle.load(open("words.pkl", 'rb'))
    classes = pickle.load(open("classes.pkl", 'rb'))
    model = load_model('chatbot_model.h5')

    # tokenize and then lemmatize the user input. eliminate punctuation
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum()]

    # create a bag of words for the pre-preprocessed user-input
    bag = [1 if word in tokens else 0 for word in words]
    bag = np.array(bag)

    # make a prediction to categorize the user input into one tag
    result = model.predict(np.array([bag]), verbose=0)[0]

    # get results and sort them from highest probability to lowest
    results = [[index, res] for index, res in enumerate(result)]
    results.sort(key=lambda x: x[1], reverse=True)

    # return list is a list of dictionaries with the keys being the intent and probability
    # and the values being the tag and the numeric probability
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def converse(intents_list, intents_json):
    # load the tags
    tag = intents_list[0]['intent']

    # load all the intents in the intents file
    intents = intents_json['intents']
    for intent in intents:
        # if the tag matches, print a random response from that tag
        # however, if the tag is 'yes', print a blank line
        # because the 'yes' responses are handled in the the guess_sign() method
        if intent['tag'] == tag:
            if tag == 'yes':
                return ''
            results = random.choice(intent['response'])
    return results


def introduce():
    # open the user model
    with open('user_model.pickle', 'rb') as handle:
        user_model = pickle.load(handle)

    # introduce star
    name = input("hi... my name is star. what is your name?\n")

    # if star hasn't seen the user before. have her reveal her functionalities
    if name not in user_model:
        info = ['']
        user_model[name] = info
        print(
            "hello, " + name + "! i don't believe we've met before. i am an AI astrology chatbot that can guess your zodiac sign or tell you random facts about the different zodiac signs. how can i help you today?")

    # if star has seen the user before....
    else:
        info = user_model[name]
        # but star doesn't know the user's zodiac sign, give a general response
        if info[0] == '':
            print("welcome back, " + name + "! how can i help you today?")

        # but star does know the user's zodiac sign, tell the user that star remembers
        else:
            print("welcome back, " + name + "! i remember that you're a " + info[0] + ". how can i help you today?")
    return name, user_model

def guess_sign(user_model, name, personality):
    # holds user's personality traits with personality_list[0] being their sign
    personality_list = user_model[name]

    # read the df holding the csv file, the model, and the vectorizer pkl files
    df = pd.read_pickle("df.pkl")
    prediction_model = pickle.load(open('sign_prediction_model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

    # vectorize the user input for what their personality is like
    new_tfidf = tfidf.transform([personality])

    # make a prediction & use the df to reverse integer encode the prediction
    prediction = prediction_model.predict(new_tfidf)
    guessed_sign = df.loc[df['sign_id'] == prediction[0]].iloc[0].loc['sign']
    print("hmmm... based on my calculations, you are a " + guessed_sign + ". am i correct?")

    # get user input and predict what they're saying
    user_input = input('')
    intents = predict_class(user_input)

    # if the user said yes, then print the appropriate response
    # based on their star sign and save their star sign into the user model
    # else print the appropriate response
    if intents[0]['intent'] == 'yes':
        intents_list = intents_file['intents']
        yes = intents_list[3]
        responses = yes['response']
        print(responses[prediction[0]])

        personality_list[0] = guessed_sign
        user_model[name] = personality_list
    else:
        response = converse(intents, intents_file)
        print(response)

    # while star does not guess correctly, keep asking for more input
    # and keep guessing
    while intents[0]['intent'] == 'no':
        user_input = input('')
        personality_list.append(user_input)

        new_tfidf = tfidf.transform([user_input])

        prediction = prediction_model.predict(new_tfidf)
        guessed_sign = df.loc[df['sign_id'] == prediction[0]].iloc[0].loc['sign']
        print("further calculations reveal that you are a " + guessed_sign + ". am i correct?")

        user_input = input('')
        intents = predict_class(user_input)

        if intents[0]['intent'] == 'yes':
            intents_list = intents_file['intents']
            yes = intents_list[3]
            responses = yes['response']
            print(responses[prediction[0]])

            personality_list[0] = guessed_sign
            user_model[name] = personality_list
            break

        response = converse(intents, intents_file)
        print(response)

    return name, user_model


def main():
    # have star introduce herself
    name, user_model = introduce()

    # holds user's personality traits with personality_list[0] being their sign
    personality_list = user_model[name]

    # get user input, predict what to say next, then say it
    user_input = input('')
    intents = predict_class(user_input)
    response = converse(intents, intents_file)
    print(response)

    # if the user asks star to predict what their star sign is
    # guess the user's star sign
    if intents[0]['intent'] == 'predict':
        personality = input()
        personality_list.append(personality)
        guess_sign(user_model, name, personality)

    # keep conversing with the user until they say good bye
    while intents[0]['intent'] != 'bye':

        # get user input, predict what to say next, then say it
        user_input = input('')
        intents = predict_class(user_input)
        response = converse(intents, intents_file)
        print(response)

        # if the user asks star to predict what their star sign is
        # guess the user's star sign
        if intents[0]['intent'] == 'predict':
            personality = input()
            personality_list.append(user_input)
            guess_sign(user_model, name, personality)

    # dump user model into a pkl file so that the data persists
    with open('user_model.pickle', 'wb') as handle:
        pickle.dump(user_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

main()


