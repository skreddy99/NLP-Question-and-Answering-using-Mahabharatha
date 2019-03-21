import random

__author__ = 'thirumal'

import os
import re
import nltk
import pickle
import urllib.request as urlrequest
import urllib.error as urlerr
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter

# initialized variables like filenames, paths
TRAINDATA_FNAME = 'traindata.txt'
TESTDATA_FNAME = 'testdata.txt'
VOCAB_FNAME = 'vocab.pkl'
indextoword_filename = 'index2word.pkl'
wordtoindex_filename = 'word2index.pkl'

tokenizer = RegexpTokenizer(r'@?\w+')
cachedStopWords = stopwords.words("english")
TRAIN_CONTEXT_PREFIX = 'train_context_with_question'
TEST_CONTEXT_PREFIX = 'test_context_with_question'
UNKNOWN_TOKEN = "UNK"


def downloadRawData(test=False):
    """
    this method fetches all sections of the 1st chapter of epic 'Ramayana' from the url, converts into a dctionary of
    section number to content. The dictionary is dumped into a pickle file called ramayana.pkl.
    :return:
    """
    sections = list(range(1002, 1238))
    sections.extend(range(2001, 2080))
    sections.extend(range(3001, 3313))
    # sections.extend(range(5001, 5199))
    if not test:
        try:
            for section in sections:
                if (section > 1002 and section <= 1200):
                    targeturl = 'http://sacred-texts.com/hin/m01/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/train_' + str(section) + '.txt')
                elif (section > 2000 and section <= 2060):
                    targeturl = 'http://sacred-texts.com/hin/m02/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/train_' + str(section) + '.txt')
                elif (section > 3000 and section <= 3200):
                    targeturl = 'http://sacred-texts.com/hin/m03/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/train_' + str(section) + '.txt')
        except urlerr.URLError:
            raise

    else:
        try:
            for section in sections:
                if (section > 1200 and section <= 1238):
                    targeturl = 'http://sacred-texts.com/hin/m01/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/test_' + str(section) + '.txt')
                elif (section > 2060 and section <= 2080):
                    targeturl = 'http://sacred-texts.com/hin/m02/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/test_' + str(section) + '.txt')
                elif (section > 3200 and section <= 3313):
                    targeturl = 'http://sacred-texts.com/hin/m03/m0' + str(section) + '.htm'
                    urlrequest.urlretrieve(targeturl, data_dir + '/test_' + str(section) + '.txt')
        except urlerr.URLError:
            raise


def readandCleanData(filename, data_dir):
    """
    reads the raw html content in the file, removes all HTML tags and the tokens listed in removetokens
    :param filename: file to be read from.
    :return: a list of strings containing text of the file.
    """

    data = []
    file_path = os.path.join(data_dir, filename)
    stoptokens = ['the mahabharata', 'book', ':', 'adi', 'Parva', 'section', '\n', 'sacred texts&nbsp;\n',
                  'hinduism&nbsp;\n',
                  'mahabharata&nbsp;\n', 'index&nbsp;\n', 'previous&nbsp;\n', 'next&nbsp;\n', 'book 1', 'book 2',
                  'next', 'i','ii','iii','iv','v','vi','vii','viii','ix','x','c','p.', '\n', ' \n']
    with open(file_path, 'r') as f:
        for line in f:
            cleanr = re.compile('<.*?>')
            text = re.sub(cleanr, '', line)
            if text.lower() not in stoptokens:
                data.append(text)

    return data


def delete_rawdata_files(test=False):
    """
    this utility method will hard delete the file from the disk given the directory path and filename.
    :param filename: name of file to be deleted
    :param data_dir: directory which contains the file
    :return: nothing
    """
    if not test:
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if filename.endswith('.txt') and filename.startswith('train_'):
                try:
                    os.remove(filepath)

                except FileNotFoundError:
                    raise
        print('deleted raw train data files from' + data_dir)

    else:
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if filename.endswith('.txt') and filename.startswith('test_'):
                try:
                    os.remove(filepath)

                except FileNotFoundError:
                    raise
        print('deleted raw test data files from' + data_dir)


def readFile(data_dir, filename):
    """

    :param filepath: path to the directory that contains the file
    :param filename: name of file to be read
    :return: a list of strings containing data from the file.
    """
    file_path = os.path.join(data_dir, filename)
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)

    return data


def writetoFile(content, filename):
    """
    writes the content to the file mentioned in filename. If the file exists in the directory path and contains data, it will be overwritten.
    :param content: content to be written to file
    :param filename: full path of the file to which it's written
    :return: nothing
    """
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'w+') as f:
        for data in content:
            for d in data:
                f.write(d)


def preprocessRawHTMLData(test=False):
    """
    This method will call readData to preprocess and writeToFile method to write the preprocessed file.
    :param data_dir: path to the directory which has all the text files to be preprocessed.
    :return: the cleaned up and preprocessed data in a new file called traindata.txt
    """
    if not test:
        traindata = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt") and filename.startswith("train_"):
                data = readandCleanData(filename, data_dir)
                traindata.append(data)

        return traindata

    else:
        testdata = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt") and filename.startswith("test_"):
                data = readandCleanData(filename, data_dir)
                testdata.append(data)

        return testdata


def convert_data_into_sentences(filename):
    """
    converts a text file into a list of sentences using nltk sentence tokenizer.
    :param filename:
    :return: a list of sentences
    """
    filepath = os.path.join(data_dir, filename)
    with open(filepath) as fp:
        data = fp.read()

    sent = sent_tokenize(data, 'english')

    return sent


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = tokenizer.tokenize(sentence)
    return [w for w in words if w not in stopwords.words("english")]


def split_into_context_question(sentences, prefix):
    """

    :param sentences: data file which is preprocessed that will be used to create question for every 20 lines.
    :return X,q,a: X - numpy array of size (len(words in context), vocabsize)
                   q - numpy array of size (len(question), vocabsize)
                   a - numpy array of size (len(answer), vocabsize)
    """
    context = []
    c_count = 0
    fpath = os.path.join(data_dir, prefix)

    # for i, sent in enumerate(sentences):
    #     if (i % 20 == 0 and i > 0 and len(context) > 0 and len(context) == 20):
    #         c = ' '.join(s.replace('\n', ' ') for s in context)
    #         question = sentences[i + 1]
    #         nnp_words = performPOS(question)
    #         if (len(nnp_words) > 0):
    #             replacewords = [str(word[0]) for word in nnp_words]
    #             # n = random.randint(len(replacewords),1)
    #             newq = question.replace(replacewords[0], '@placeholder')
    #
    #             with open(fpath + str(c_count) + '.txt', 'w+') as f:
    #                 f.write(c + '\n')
    #                 f.write('\n' + newq + '\n')
    #                 f.write('\n' + replacewords[0])
    #
    #             context = []
    #             c_count += 1
    #         else:
    #             context = []
    #     if (i % 21 == 0 and i > 0):
    #         continue
    #     elif (len(context) < 20):
    #         context.append(sent)

    ctx_len = 10 # Length of each context file
    for i,sentence in enumerate(sentences):
        n_context_files = (len(sentences) // 2)
        if(i<n_context_files-ctx_len+2):
            c = sentences[i:i+ctx_len]
            context = ' '.join(s.replace('\n', ' ') for s in c)

            question = sentences[i+ctx_len+1]
            nnp_words = performPOS(question)
            if (len(nnp_words) > 0):
                replacewords = [str(word[0]) for word in nnp_words]
                newq = question.replace(replacewords[0], '@placeholder')
                with open(fpath + str(i) + '.txt', 'w+') as f:
                    f.write(context + '\n')
                    f.write('\n' + newq + '\n')
                    f.write('\n' + replacewords[0])

def performPOS(sentence):
    """
    :param sentence: a string
    :return: list of words whose POS is tagged as 'NNP'
    """
    sent_list = basic_tokenizer(sentence)
    if (len(sent_list) > 0):
        tagged_sent = nltk.pos_tag(sent_list)
        NNP_words = []
        for sent in tagged_sent:
            for tag in sent:
                if ('NNP' in tag):
                    NNP_words.append(sent)
                elif ('NN' in tag):
                    NNP_words.append(sent)

        return list(NNP_words)

    print('sentence was empty, can\'t perform NER')
    return


def create_vocab(sentences, vocab_size):
    """
    creates a vocab of size lesser than or equal to vocab_size. vocab is just a list of unique words in the raw data.
    :param sentences:
    :param vocab_size:
    :return:
    """
    word_freq = Counter()
    vocab = []
    word_to_index = dict()
    index_to_word = dict()

    # tokenize all sentences
    tokenized_sentences = [basic_tokenizer(sent) for sent in sentences]

    # get the word frequency for each word in the data.txt
    for sent in tokenized_sentences:
        for w in sent:
            word_freq[w] += 1

    print("Found %d unique words tokens." % len(word_freq.items()))

    # create a vocabulary of vocab_size with words whose frequency is > 1 and < 2000.
    for k, v in word_freq.items():
        if v >= 10 and v < 1000 and len(vocab) <= vocab_size:
            vocab.append(k.lower())

    vocab.append('UNK')
    vocab.append('<DELIMITER>')
    vocab.append('@placeholder')
    print("vocabulary size is %d." % len(list(set(vocab))))
    sorted_vocab = list(set(sorted(vocab)))

    # create word_to_index and index_to_word for lookup
    for i, w in enumerate(sorted_vocab):
        word_to_index[w] = i
        index_to_word[i] = w
        i += 1

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in vocab]

    return list(set(sorted_vocab)), word_to_index, index_to_word


def create_vectors(fin):
    """
    :param fin: input filename
    :return:
    """
    vocab = pickle.load(open(os.path.join(data_dir, VOCAB_FNAME), 'rb'))
    word_to_index = pickle.load(open(os.path.join(data_dir, wordtoindex_filename), 'rb'))

    data = readFile(data_dir, fin)

    context = data[0]
    question = data[2]
    answer = data[4]

    # convert words to ids
    context_sentences = sent_tokenize(context)
    context_tokens = [basic_tokenizer(sentence) for sentence in context_sentences]
    question_tokens = basic_tokenizer(question)

    for i, sentence in enumerate(context_tokens):
        for j, word in enumerate(sentence):
            if (word not in vocab):
                sentence[j] = word_to_index['UNK']
            else:
                sentence[j] = word_to_index[word.lower()]

    for i, word in enumerate(question_tokens):
        if (word not in vocab):
            question_tokens[i] = word_to_index['UNK']
        else:
            question_tokens[i] = word_to_index[word.lower()]

    # if the answer term is not in vocab, assign UNK as the answer
    if answer.lower() not in vocab:
        answer_token = word_to_index['UNK']
    else:
        answer_token = word_to_index[answer.lower()]

    context_words_len = sum(len(s) for s in context_tokens)

    X = np.zeros((context_words_len, len(vocab)))
    q = np.zeros((len(question_tokens), len(vocab)))
    a = np.zeros((1, len(vocab)))

    for i, word in enumerate(context_tokens):
        X[i, word] = 1

    for j, word in enumerate(question_tokens):
        q[j, word] = 1

    a[0, answer_token] = 1

    X.reshape(context_words_len, len(vocab))
    q.reshape(len(question_tokens), len(vocab))
    a.reshape(1, len(vocab))

    return X, q, a


def prepare_data(test=False):
    """
    main method that will call other helper methods to download the raw data and transform into
    context,query,answer vectors of desired shape and dump in pickle files.
    :return:
    """
    # if data is not already downloaded, then download the data from web, remove HTML tokens,
    # aggregate all files in data.txt and delete all raw downloaded files.
    if not test:
        if not os.path.isfile(os.path.join(data_dir, TRAINDATA_FNAME)):
            print('downloading files as they are not present')
            downloadRawData()
            print('files downloaded')
            data = preprocessRawHTMLData(test=False)
            print('data preprocessed, html tags removed...')
            writetoFile(data, TRAINDATA_FNAME)
            print('preprocessed data is written to traindata.txt in ' + data_dir)
            delete_rawdata_files(test=False)
            print('raw files deleted')
        else:
            print('skipped downloading and html preprocessing, traindata.txt already available')

        rawdata_sentences = convert_data_into_sentences(os.path.join(data_dir, TRAINDATA_FNAME))

        if not os.path.isfile(os.path.join(data_dir, VOCAB_FNAME)):
            # create vocabulary from rawdata
            vocab, word_to_index, index_to_word = create_vocab(sentences=rawdata_sentences, vocab_size=1000)

        if not os.path.isfile(os.path.join(data_dir, wordtoindex_filename)):
            pickle.dump(word_to_index, open(os.path.join(data_dir, wordtoindex_filename), 'wb'))
            pickle.dump(index_to_word, open(os.path.join(data_dir, indextoword_filename), 'wb'))
            pickle.dump(vocab, open(os.path.join(data_dir, VOCAB_FNAME), 'wb'))

        # create context and question files from data.txt..
        split_into_context_question(rawdata_sentences, TRAIN_CONTEXT_PREFIX)

        for filename in os.listdir(data_dir):
            if filename.startswith("train_context") and filename.endswith('.txt'):
                X, q, a = create_vectors(filename)
                fX = os.path.join(data_dir, filename[:-4] + '_ctx')
                fq = os.path.join(data_dir, filename[:-4] + '_q')
                fa = os.path.join(data_dir, filename[:-4] + '_a')
                pickle.dump(X, open(fX, 'wb'))
                pickle.dump(q, open(fq, 'wb'))
                pickle.dump(a, open(fa, 'wb'))
    else:
        if not os.path.isfile(os.path.join(data_dir, TESTDATA_FNAME)):
            print('downloading test files as they are not present')
            downloadRawData(test=True)
            print('files downloaded')
            data = preprocessRawHTMLData(test=True)
            print('data preprocessed, html tags removed...')
            writetoFile(data, TESTDATA_FNAME)
            print('preprocessed data is written to data.txt in ' + data_dir)
            delete_rawdata_files(test=True)
            print('raw test files deleted')
        else:
            print('skipped downloading and html preprocessing, testdata.txt already available')

        test_rawdata_sentences = convert_data_into_sentences(os.path.join(data_dir, TESTDATA_FNAME))

        # create context and question files from data.txt..
        split_into_context_question(test_rawdata_sentences, TEST_CONTEXT_PREFIX)

        for filename in os.listdir(data_dir):
            if filename.startswith("test_context") and filename.endswith('.txt'):
                X, q, a = create_vectors(filename)
                fX = os.path.join(data_dir, filename[:-4] + '_ctx')
                fq = os.path.join(data_dir, filename[:-4] + '_q')
                fa = os.path.join(data_dir, filename[:-4] + '_a')
                pickle.dump(X, open(fX, 'wb'))
                pickle.dump(q, open(fq, 'wb'))
                pickle.dump(a, open(fa, 'wb'))


def create_data_directory():
    """
    creates a 'data/' directory at the same level as 'code/'
    This directory will be used to download and write all data files in this project.
    :return:
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    return data_dir


if __name__ == '__main__':
    data_dir = create_data_directory()
    prepare_data(test=False)
    prepare_data(test=True)
    # create_vectors('train_context_with_question534.txt')