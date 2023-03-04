import pickle
import nltk
from nltk import *


def unpickle(filename):
    with open(filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data


def compute_probability(text, unigram_dict, bigram_dict, v):
    unigrams_test = word_tokenize(text)
    bigrams_test = list(ngrams(unigrams_test, 2))

    p_laplace = 1

    for bigram in bigrams_test:
        b = bigram_dict[bigram] if bigram in bigram_dict else 0  # b is the bigram count
        u = unigram_dict[bigram[0]] if bigram[0] in unigram_dict else 0  # u is the unigram count of the first word in the bigram
        p_laplace = p_laplace * ((b + 1) / (u + v))

    return p_laplace


def main():
    # output file to hold the predictions
    predictions_filename = "predictions.txt"
    predictions_file = open(predictions_filename, "w+")

    # unpicking dictionaries
    english_unigram_dict = unpickle("english_unigram_dict.pickle")
    english_bigram_dict = unpickle("english_bigram_dict.pickle")

    french_unigram_dict = unpickle("french_unigram_dict.pickle")
    french_bigram_dict = unpickle("french_bigram_dict.pickle")

    italian_unigram_dict = unpickle("italian_unigram_dict.pickle")
    italian_bigram_dict = unpickle("italian_bigram_dict.pickle")

    # v is the total vocabulary size (add the lengths of the 3 unigram dictionaries)
    v = len(english_unigram_dict) + len(french_unigram_dict) + len(italian_unigram_dict)

    # open the test data file and read in the test data
    test_filename = "LangId.test"
    with open(os.path.join(os.getcwd(), "data", test_filename), mode='r', encoding="utf-8") as f:
        lines = f.readlines()

    # calculate the probabilities of each line being english, french, or italian
    # and write the most likely language to an output file
    for line in lines:
        english_prob = compute_probability(line, english_unigram_dict, english_bigram_dict, v)
        french_prob = compute_probability(line, french_unigram_dict, french_bigram_dict, v)
        italian_prob = compute_probability(line, italian_unigram_dict, italian_bigram_dict, v)

        highest_prob = max(english_prob, french_prob, italian_prob)

        if highest_prob == english_prob:
            predictions_file.write("English\n")
        elif highest_prob == french_prob:
            predictions_file.write("French\n")
        elif highest_prob == italian_prob:
            predictions_file.write("Italian\n")

    # opens the predictions files and reads predictions into a list
    predictions_file.seek(0, 0)  # set 'pointer' to beginning of file
    with open(predictions_filename, mode='r+', encoding="utf-8") as f:
        predictions = f.readlines()

    for i in range(len(predictions)):
        predictions[i] = predictions[i].replace("\n", "")

    # open the solutions file and read in the solutions
    solutions_filename = "LangId.sol"
    with open(os.path.join(os.getcwd(), "data", solutions_filename), mode='r', encoding="utf-8") as f:
        solutions = f.readlines()

    # count correct classifications & output if the model
    # incorrectly classified text
    correct = 0
    for solution in solutions:
        line_number, language = solution.split(" ")
        language = language.replace("\n", "")
        line_number = int(line_number)
        if language == predictions[line_number - 1]:
            correct += 1
        else:
            print("\nIncorrect prediction at line ", line_number)
            print("Text:", lines[line_number - 1].replace("\n", ""))
            print("The model predicted", predictions[line_number - 1], "but the correct language is", language)

    print("\nTotal accuracy of language prediction model: {:0.2f}%".format((correct / len(solutions)) * 100))


if __name__ == '__main__':
    main()
