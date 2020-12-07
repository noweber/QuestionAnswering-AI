import nltk
import os
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    # TODO: use os.listdir and functions similar to traffic .. listdir() and path.join from project 5 are essential
    # TODO: you should explicitly only look for .txt files within the directory

    files = dict()

    # Find each file within the given directory:
    for filename in os.listdir(directory):
        print("filename: ", filename)
        if '.txt' in filename:
            print("text file found!")
            with open(os.path.join(directory, filename), encoding="utf8") as file:
                # file_text = [
                #     word.lower() for word in
                #     nltk.word_tokenize(file.read())
                #     if word.isalpha()
                # ]
                file_text = file.read()
                print("file_text length: ", len(file_text))
                files[filename] = file_text
    
    print("files count: ", len(files))
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # TODO: nltk has functions to remove punctuation and stop words
    # TODO: tokenize will help solve the third example from the spec
    # TODO: definitely need to use .tolower
    
    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # TODO: look at the lecture notes for how to calculate the idf function
    # TODO: brian provides the code for this
    # TODO: make sure to use base 'e lg for this
    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # TODO: use python sort() which lets you pass words in.. this will make it more Pythonic .. else it will just be normal code
    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # TODO: use .sorted() with helper function of keys??
    raise NotImplementedError


if __name__ == "__main__":
    main()
