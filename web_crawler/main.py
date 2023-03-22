import nltk
from nltk import *
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from bs4 import BeautifulSoup

import requests
import pickle
import re
import os
import lxml


def crawl(original_url):
    # holds the urls gathered from the original link
    urls_list = [original_url]

    # a queue of which link to visit next & a list of which links have already been visited for BFS
    url_queue = [original_url]
    visited = []

    broken_links = ['https://www.cnbc.com/2022/12/15/google-vs-chatgpt-what-happened-when-i-swapped-services-for-a-day.html']

    # regex used to validate urls
    regex = ("((http|https)://)(www.)?" +
             "[a-zA-Z0-9@:%._\\+~#?&//=]" +
             "{2,256}\\.[a-z]" +
             "{2,6}\\b([-a-zA-Z0-9@:%" +
             "._\\+~#?&//=]*)")

    while url_queue and len(urls_list) <= 15:
        # current link
        link_to_visit = url_queue[0]
        del url_queue[0]

        # using requests to access the page
        f = requests.get(link_to_visit)

        # beautiful soup creates a soup object of the page
        soup = BeautifulSoup(f.content, 'lxml')

        # validate url using regex, then put it in the url list
        for link in soup.find_all('a'):
            link = str(link.get('href'))
            # eliminates social media
            if "facebook" in link or "twitter" in link or "youtube" in link or "linkedin" in link or "reddit" in link:
                continue
            # eliminates photos
            if "jpeg" in link:
                continue
            # eliminates pdfs
            if "pdf" in link:
                continue
            # eliminates duplicates from archives
            if "archive" in link:
                continue
            # release notes aren't that helpful
            if "release-notes" in link:
                continue
            if "chatgpt" in link and link not in visited and link not in broken_links:
                match = re.search(regex, link)
                if match:
                    urls_list.append(link)
                    url_queue.append(link)
                    visited.append(link)

    return urls_list[:25]


def scrape(url, number):
    # using requests to access the page
    f = requests.get(url, headers={"User-Agent": "XY"})

    # beautiful soup creates a soup object of the page
    soup = BeautifulSoup(f.content, 'lxml')

    # extracting text from the website
    text = soup.get_text()

    # writes text to a file in the scraped text directory
    with open(os.path.join(os.getcwd(), "scraped_text", f'link{number}_data.txt'), mode='w') as f:
        print("Writing file...", number)
        f.write(text)
        print("Finished writing file", number)


def preprocess_text():
    number = 0
    directory = 'scraped_text'
    for filename in os.listdir(directory):
        r = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(r):
            # read the content of the file as raw text
            with open(r) as f:
                raw_text = f.read()

            # remove new lines and tabs
            raw_text = raw_text.replace('\n', ' ')
            raw_text = raw_text.replace('\t', ' ')

            # extracts sentences with NLTk's sentence tokenizer
            sentences = sent_tokenize(raw_text)

            # write sentences to corresponding file
            with open(os.path.join(os.getcwd(), "cleaned_data", f'link{number}_data.txt'), mode='w') as f:
                for sentence in sentences:
                    f.write(sentence)
                    f.write("\n")

        number += 1


def find_important_terms():
    stop = set(stopwords.words('english'))

    # manual stopwords... either big news sources or just odd terms i found
    custom_stop = ['insider', 'axios', 'npr', 'sie', 'cnn', 'january', 'february', 'december', 'cnbc', 'isbn', 'techcrunch'
                   'guardian', 'pmid', 'govtech', 'verge', 'gizmodo', 'mashable', 'vox', 'issn', 'forbes', 'zdnet', 'cbc', 'slate',
                   'reuters', 'eweek', 'livemint', 'engadget', 'technologyadvice', 'atlantic', 'digiday', 'jo', 'yorker', 'august', 'times'
                   'variety', 'march', 'tass']

    corpus = []

    directory = 'cleaned_data'
    for filename in os.listdir(directory):
        r = os.path.join(directory, filename)

        if os.path.isfile(r):
            with open(r) as f:
                raw_text = f.read()

            sentences = raw_text.split("\n")

            # add each sentence to corpus after cleaning it up
            for sentence in sentences:
                word_tokens = word_tokenize(sentence)
                filtered = [word.lower() for word in word_tokens if
                            word.lower() not in stop
                            and word.lower().isalpha()
                            and word.lower() not in custom_stop]
                sentence = ' '.join(filtered)
                corpus.append(sentence)

    # tf-idf to find most important terms
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    print("most frequent terms:")
    print(df.max().sort_values(ascending=False)[:50])

    return corpus

def create_knowledge_base(corpus, list):
    # knowledge base in the form:
    # {term: [list of phrases containing word]}
    knowledge_base = {}
    for word in list:
        phrase_list = []
        for phrase in corpus:
            if word in phrase:
                phrase_list.append(phrase)
        knowledge_base[word] = phrase_list

    with open('knowledge_base.pkl', 'wb') as handle:
        pickle.dump(knowledge_base, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return knowledge_base


def main():
    url_list = crawl('https://en.wikipedia.org/wiki/ChatGPT')
    print(url_list)

    count = 0
    for link in url_list:
        scrape(link, count)
        count += 1

    preprocess_text()
    corpus = find_important_terms()

    chosen_terms = ["chatgpt", "openai", "conversation", "robotics", "limitations", "chat", "math", "science", "math", "write"]
    knowledge_base = create_knowledge_base(corpus, chosen_terms)
    print(knowledge_base)


if __name__ == '__main__':
    main()
