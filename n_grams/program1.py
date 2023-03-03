import os
import pickle
import nltk
from nltk import *


def create_ngrams_dict(filename):
    try:
        # opens file and reads in raw text
        with open(os.path.join(os.getcwd(), "data", filename), mode='r', encoding="utf-8") as f:
            raw_text = f.read()

        # removes newlines
        raw_text = raw_text.replace('\n', ' ')

        # tokenizes the raw text
        tokens = word_tokenize(raw_text)

        # creates a unigrams list
        unigrams = tokens

        # creates a bigrams list
        bigrams = list(ngrams(unigrams, 2))

        # creates a unigrams dictionary of unigrams and counts
        unigram_dict = {t: unigrams.count(t) for t in set(unigrams)}

        # creates a bigrams dictionary of bigrams and counts
        bigram_dict = {b: bigrams.count(b) for b in set(bigrams)}

        return unigram_dict, bigram_dict

    except FileNotFoundError:
        print("Error: File path not specified correctly")
        exit()


def main():
    training = {"english": "LangId.train.English",
                "french": "LangId.train.French",
                "italian": "LangId.train.Italian"}

    for language, file in training.items():
        unigram_dict, bigram_dict = create_ngrams_dict(file)

        uni_file = language + "_unigram_dict.pickle"
        bi_file = language + "_bigram_dict.pickle"

        # pickles unigram dictionaries
        with open(uni_file, 'wb') as handle:
            pickle.dump(unigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # pickles bigrams dictionaries
        with open(bi_file, 'wb') as handle:
            pickle.dump(bigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
