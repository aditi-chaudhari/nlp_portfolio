import sys
import nltk
from nltk import *
from nltk.corpus import stopwords
import random


def preprocess_text(tokens):
    processed = []
    stops = set(stopwords.words('english')) # stopwords

    # ensures that tokens in the processed list are alphabetic, not in the stopwords list
    # and are greater than length 5
    for token in tokens:
        if token.isalpha() and token not in stops and len(token) > 5:
            processed.append(token.lower())

    # lemmatizes the words
    lemmatizer = nltk.WordNetLemmatizer()
    token_lemmas = [lemmatizer.lemmatize(token) for token in processed]

    # finds the unique lemmas
    unique_token_lemmas = set(token_lemmas)

    # pos tags the unique lemmas
    tagged = nltk.pos_tag(unique_token_lemmas)

    # finds and prints the first 20 lemmas
    first_twenty = []
    for i in range(20):
        first_twenty.append(tagged[i])
    print("The first 20 tagged lemmas are:", first_twenty)

    # finds nouns in pos tagged lemmas
    nouns = [token[0] for token in tagged if token[1][0] == 'N']

    print("The number of tokens in the text is: {}".format(len(processed)))
    print("The number of nouns in the text is: {}".format(len(nouns)))

    return processed, nouns


def word_guessing_game(word_list, score):
    # randomly chooses a word to guess
    word_to_guess = word_list[random.randint(0, 49)]
    word_length = len(word_to_guess)
    reveal = ['_'] * word_length

    # maps each letter in the word to guess to what index
    # it can be found in the word to guess
    answer = {}
    for position, letter in enumerate(word_to_guess):
        if letter not in answer:
            lst = [position]
            answer[letter] = lst
        else:
            lst = answer[letter]
            lst.append(position)
            answer[letter] = lst

    print("Let's play a word guessing game!")

    while True:
        # if the score is negative, the user lost
        if score < 0:
            print("Sorry! You lost! The word was ", word_to_guess)
            break

        # prints the word to guess with blanks
        for element in reveal:
            print(element, end=" ")

        # makes user guess a letter
        guessed_letter = input("\nGuess a letter: ").lower()

        # if the guessed letter is in the word to guess,
        # increment score
        # update reveal list
        # delete the guessed letter from the dictionary
        if guessed_letter in answer:
            score += 1
            print("Right! Score is ", score)
            lst = answer[guessed_letter]
            for element in lst:
                reveal[element] = guessed_letter
            del answer[guessed_letter]
        # else decrement score
        else:
            score -= 1
            print("Sorry! Incorrect guess. Score is ", score)

        if '_' not in reveal:
            print("You guessed it!")
            break

    return score


if __name__ == '__main__':
    # check if user inputted a file name & print an error if the user didn't
    if len(sys.argv) < 2:
        print("Error: File name not specified.")
        exit()
    else:
        file_name = sys.argv[1]

        # read the content of the file as raw text
        with open(file_name) as f:
            raw_text = f.read()

        # tokenizes the raw text
        tokens = word_tokenize(raw_text)

        # calculates lexical diversity (formula from https://www.nltk.org/book/ch01.html)
        ld = len(set(tokens)) / len(tokens)
        print(
            'The text has a lexical diversity of {:0.2f}, which means that {:.0f}% of the words in the text are unique'.format(
                ld, ld * 100))

        # pre-process text
        processed_tokens, processed_nouns = preprocess_text(tokens)

        # create a dictionary of {noun: number of occurrences of that noun in the tokens}
        noun_count = {}
        for noun in processed_nouns:
            noun_count[noun] = processed_tokens.count(noun)

        # sorts dictionary
        noun_count = dict((sorted(noun_count.items(), key=lambda item: item[1], reverse=True)))

        # creates a list of the most common nouns
        noun_count_keys = list(noun_count.keys())

        # selects the 50 most common nouns
        fifty_most_common = []
        for i in range(50):
            fifty_most_common.append(noun_count_keys[i])
        print("The 50 most common nouns are: ", fifty_most_common)

        # runs word guessing game
        score = 5
        score = word_guessing_game(fifty_most_common, score)
        while input("Would you like to play again? [y/n]: ").lower() == 'y':
            if score < 0:
                score = 5
            word_guessing_game(fifty_most_common, score)
        print("Goodbye!")
