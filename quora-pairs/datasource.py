import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn import feature_extraction as fe
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath("../"))

from utils import memoize, log
import processing as p
import extraction as e


def load_data(raw=True):
    return load_train_data(raw), load_test_data(raw)


def load_train_data(raw=False):
    log('Loading train data...')
    data = pd.read_csv('input/train.csv')
    data = inputing_data(data)
    if raw is True:
        return data

    data['shared_words'] = load_words_data(data)
    data['shared_chars'] = load_chars_data(data)
    data['count'] = load_count_data(data)
    data['similar'] = load_similar_data(data)

    qs = questions(data)
    data['freqs1'] = load_freqs1_data(data, qs)
    data['freqs2'] = load_freqs2_data(data, qs)

    # encoder = questions_encoder(train_data, test_data)
    # x_q1 = e.extract(data, 'question1', encoder)
    # data['tfidf'] = load_freqs_data(data, freqs)

    log('DONE train data.')
    return data


def load_test_data(raw=False):
    log('Loading test data...')
    data = pd.read_csv('input/test.csv')
    data = inputing_data(data)
    if raw is True:
        return data

    data['shared_words'] = load_words_data(data)
    data['shared_chars'] = load_chars_data(data)
    data['count'] = load_count_data(data)
    data['similar'] = load_similar_data(data)

    qs = questions(data)
    data['freqs1'] = load_freqs1_data(data, qs)
    data['freqs2'] = load_freqs2_data(data, qs)

    log('DONE test data.')
    return data


def inputing_data(data):
    log('Inputing data...')
    return data.fillna('')


def normalized_chars_share(row):
    c1 = set(list(row.question1))
    c2 = set(list(row.question2))
    return len(c1 & c2) / (len(c1) + len(c2))


def load_chars_data(data):
    log('Loading shared chars...')
    return data.apply(normalized_chars_share, axis=1, raw=True)


def normalized_count(row):
    w1 = len(row.question1.split())
    w2 = len(row.question2.split())
    return min(w1, w2) / max(w1, w2)


def load_count_data(data):
    log('Loading count...')
    return data.apply(normalized_count, axis=1, raw=True)


def normalized_similar(row):
    return SequenceMatcher(None, row.question1, row.question2).ratio()


def load_similar_data(data):
    log('Loading similar...')
    return data.apply(normalized_similar, axis=1, raw=True)


def normalized_tfidf(row, x_q1, x_q2):
    tf1 = x_q1.iloc[row.id].sum()
    tf2 = x_q2.iloc[row.id].sum()
    return min(tf1, tf2) / max(tf1, tf2)


# def load_tdidf_data(data):
# return data.apply(normalized_tfidf, x_q1=x_q1, x_q2=x_q2, axis=1,
# raw=True)


def questions_corpus(train_data, test_data):
    data = train_data.question1.append(train_data.question2)
    return data.append(test_data.question1).append(test_data.question2).values


def questions_encoder(train_data, test_data):
    corpus = questions_corpus(train_data, test_data)
    question_we = p.word_encoder(
        corpus, title='corpus', stop_words=[], max_features=500)
    question_features = question_we.get_feature_names()
    return question_we, question_features


stops = fe.text.ENGLISH_STOP_WORDS


def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(),
                 row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(),
                 row['question2'].split(" ")))
    return len(w1 & w2) / (len(w1) + len(w2))


def shared_words(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are
        # nothing but stopwords
        return 0
    shared_words = [w for w in q1words.keys() if w in q2words]
    R = (len(shared_words)) / (len(q1words) + len(q2words))
    return R


def load_words_data(data):
    log('Loading shared words...')
    return data.apply(shared_words, axis=1, raw=True)


#
# frequencies
#
def questions(data):
    q = data.question1.append(data.question2)
    return pd.DataFrame(q.value_counts())


def freq1(row, questions):
    return questions.loc[row['question1']][0]


def freq2(row, questions):
    return questions.loc[row['question2']][0]


def load_freqs1_data(data, questions):
    log('Loading freqs1...')
    return data.apply(freq1, questions=questions, axis=1, raw=True)


def load_freqs2_data(data, questions):
    log('Loading freqs2...')
    return data.apply(freq2, questions=questions, axis=1, raw=True)


#
# n-grams
#
stops = set(stopwords.words('english'))
color = sns.color_palette()


def get_stops_ratio(row):
    q1stops = set(row["stops_ques1"])
    return float(max(len(q1stops), 0)) / max(len(set(q1stops.union(row["stops_ques2"]))), 1)


def load_stops(data):
    log('Loading stops...')
    data["stops_ques1"] = data['question1'].apply(get_stops)
    data["stops_ques2"] = data['question2'].apply(get_stops)
    data["stops_ratio"] = data.apply(get_stops_ratio, axis=1)


def get_words(que):
    # TODO: try stemming and lemminga
    return word_tokenize(str(que).lower())


def get_stops(que):
    return [word for word in get_words(que) if word in stops]


def get_unigrams(que):
    return [word for word in get_words(que) if word not in stops]


def get_common_unigrams(row):
    return len(set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])))


def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count"]) / max(len(set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"]))), 1)


def load_unigrams(data):
    log('Loading unigrams...')
    data["unigrams_ques1"] = data['question1'].apply(get_unigrams)
    data["unigrams_ques2"] = data['question2'].apply(get_unigrams)
    data["unigrams_common_count"] = data.apply(get_common_unigrams, axis=1)
    data["unigrams_common_ratio"] = data.apply(
        get_common_unigram_ratio, axis=1)


def get_bigrams(que):
    return [i for i in ngrams(que, 2)]


def get_common_bigrams(row):
    return len(set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])))


def get_common_bigram_ratio(row):
    return float(row["bigrams_common_count"]) / max(len(set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"]))), 1)


def load_bigrams(data):
    log('Loading bigrams...')
    data["bigrams_ques1"] = data["unigrams_ques1"].apply(get_bigrams)
    data["bigrams_ques2"] = data["unigrams_ques2"].apply(get_bigrams)
    data["bigrams_common_count"] = data.apply(get_common_bigrams, axis=1)
    data["bigrams_common_ratio"] = data.apply(get_common_bigram_ratio, axis=1)


def get_trigrams(que):
    return [i for i in ngrams(que, 3)]


def get_common_trigrams(row):
    return len(set(row["trigrams_ques1"]).intersection(set(row["trigrams_ques2"])))


def get_common_trigram_ratio(row):
    return float(row["trigrams_common_count"]) / max(len(set(row["trigrams_ques1"]).union(set(row["trigrams_ques2"]))), 1)


def load_trigrams(data):
    log('Loading trigrams...')
    data["trigrams_ques1"] = data["unigrams_ques1"].apply(get_trigrams)
    data["trigrams_ques2"] = data["unigrams_ques2"].apply(get_trigrams)
    data["trigrams_common_count"] = data.apply(get_common_trigrams, axis=1)
    data["trigrams_common_ratio"] = data.apply(get_common_bigram_ratio, axis=1)
