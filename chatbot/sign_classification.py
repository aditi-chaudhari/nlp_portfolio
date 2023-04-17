import os
import csv
import pickle

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import pandas as pd

stops = set(stopwords.words('english'))


def preprocess():
    # i defined custom stop words since a few words were repeated a lot and weren't important
    custom_stops = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio',
                    'sagittarius', 'capricorn', 'aquarius', 'pisces', 'information', 'astrology',
                    'zodiac', 'sign', 'complete', 'horoscope', 'element', 'color', 'quality', 'day'
                                                                                              'ruler', 'greatest',
                    'compatibility', 'lucky', 'numbers', 'dates', 'personality',
                    'traits', 'strengths', 'weaknesses', 'likes', 'dislikes', 'love', 'sex', 'compatibility',
                    'friends', 'family']

    # the data to dump into a csv to train a model with
    data = []

    directory = 'scraped_text'
    for filename in os.listdir(directory):
        r = os.path.join(directory, filename)
        if os.path.isfile(r):
            with open(r) as f:
                raw_text = f.read()

            # get rids of new lines and tabs
            raw_text = raw_text.replace('\n', ' ')
            raw_text = raw_text.replace('\t', ' ')

            # tokenizes sentences
            sentences = sent_tokenize(raw_text)

            with open(os.path.join(os.getcwd(), "sentences", filename), mode='w') as f:
                for sentence in sentences:
                    # we can keep whole sentences to randomly throw into our chatbot
                    f.write(sentence)
                    f.write("\n")

                    # these sentences can be further tokenized into words and then lemmatized and
                    # then filtered to train our model with
                    lemmatizer = WordNetLemmatizer()
                    tokens = word_tokenize(sentence)
                    filtered = [lemmatizer.lemmatize(word.lower()) for word in tokens if
                                lemmatizer.lemmatize(word.lower()) not in stops and lemmatizer.lemmatize(
                                    word.lower()) not in custom_stops and word.isalpha()]
                    preprocessed_sentence = ' '.join(filtered)

                    # sets up rows to be dumped into csv
                    sign = filename.replace('_data.txt', '')
                    row = [sign, preprocessed_sentence]
                    data.append(row)

    # dumps rows into csv
    with open('zodiac.csv', 'w', encoding='UTF8') as c:
        writer = csv.writer(c)
        writer.writerows(data)


def build_model():
    # reads csv
    df = pd.read_csv('zodiac.csv', names=['sign', 'text'])

    # shuffles csv to introduce randomness & better train our model
    df = df.sample(frac=1, random_state=1234)

    # integer encoding to give each zodiac sign a number
    df['sign_id'] = df['sign'].factorize()[0]

    # vectorizer
    tfidf = TfidfVectorizer(encoding='utf-8', stop_words='english')

    # extracts features and labels
    # transform the features into tfidf representation
    features = tfidf.fit_transform(df.text).toarray()
    labels = df.sign_id

    # i tried a few ml models (including naive bayes, logistic regression, random forest, etc),
    # but linear svc performed the best
    svc = LinearSVC()

    # split into an 80/20 train/test
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.20, random_state=1234)

    # train the model, then test it, and determine accuracy (about 22%)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print("\naccuracy score: ", accuracy_score(y_test, y_pred))

    # export the dataframe, the vectorizer, and the model to pickle files
    df.to_pickle("df.pkl")
    pickle.dump(tfidf, open("vectorizer.pkl", 'wb'))
    pickle.dump(svc, open("sign_prediction_model.pkl", 'wb'))


preprocess()
build_model()
