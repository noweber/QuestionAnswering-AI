from collections import Counter
import math
import nltk
from nltk.corpus import stopwords
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

    # Find each file within the given directory:
    files = dict()
    for filename in os.listdir(directory):
        if '.txt' in filename:
            with open(os.path.join(directory, filename), encoding="utf8") as file:
                file_text = file.read()
                files[filename] = file_text
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Tokenize the sentence while removing the punctuation and setting all words to lowercase:
    words = [
                word.lower() for word in
                nltk.word_tokenize(document)
                if word.isalpha()
            ]

    # Remove the English stopwords:
    # source: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    stop_words = set(stopwords.words('english'))  
    words = [word for word in words if not word in stop_words]

    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Calculate the number of documents each word appears in:
    document_frequencies = dict()
    for document in documents:
        # print("document: ", document)
        document_words = set()
        for word in documents[document]:
            if word not in document_words:
                document_words.add(word)
        for word in document_words:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1

    # Calculate the inverse document frequencies for each word:
    total_documents = len(documents)
    for key in document_frequencies:
        document_frequencies[key] = math.log(total_documents / document_frequencies[key])
    return document_frequencies


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Calculate the TF-IDFs for each word:
    term_frequency_idfs = dict()
    for document in files:
        term_frequency_idfs[document] = {}
        word_count = Counter(files[document])
        for word in word_count:
            term_frequency = word_count[word]
            term_frequency_idfs[document][word] = term_frequency * idfs[word]

    # Calculate the sum of all TF-IDFs for each word in the query by document:
    tf_idf_query_sums = []
    for document in files:
        sum_of_tf_idf_values_for_words_in_query = 0
        for word in query:
            if word in term_frequency_idfs[document]:
                sum_of_tf_idf_values_for_words_in_query += term_frequency_idfs[document][word]
        tf_idf_query_sums.append((document, sum_of_tf_idf_values_for_words_in_query))

    # Return the top 'n' documents based on the highest sum of TF-IDFs matching the query:
    tf_idf_query_sums.sort(key=lambda x:x[1], reverse=True)
    top_files = tf_idf_query_sums[:n]
    for i in range(len(top_files)):
        top_files[i] = top_files[i][0]
    return top_files


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
