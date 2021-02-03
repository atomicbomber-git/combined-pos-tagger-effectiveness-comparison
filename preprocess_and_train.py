#!/usr/bin/python

from nltk.corpus.reader.chasen import test
from nltk.tag.sequential import BigramTagger, DefaultTagger, TrigramTagger
from sklearn.model_selection import KFold
from nltk.tag import UnigramTagger
import numpy as np
from sklearn.metrics import classification_report
from pandas import DataFrame
import pickle

N_FOLD = 5

UNIGRAM_BIGRAM = "unigram_bigram"
UNIGRAM_TRIGRAM = "unigram_trigram"
UNIGRAM_BIGRAM_TRIGRAM = "unigram_bigram_trigram"

model_types = [
    UNIGRAM_BIGRAM,
    UNIGRAM_TRIGRAM,
    UNIGRAM_BIGRAM_TRIGRAM
]

corpora = {
    "corpus_1": "./corpora/corpus_1.txt"
}

def get_model_path(name, type, fold):
    return "./models/{}-{}-{}.model".format(name, type, fold)

def get_training_data_path(name, fold):
    return "./training_data/{}-{}.txt".format(name, fold)

def get_test_data_path(name, fold):
    return "./test_data/{}-{}.txt".format(name, fold)

def serizalize_corpus_data(training_data):
    return "\n".join([
        " ".join(["/".join(part) for part in line])
            for line in training_data
    ])

def pair_to_tuple(pair_text):
    token, pos = pair_text.split("/")
    return (token, pos)

def deserialize_corpus_data(lines):
    lines = [line.strip().split(" ") for line in lines]
    return [[pair_to_tuple(part) for part in line] for line in lines]

def save_corpus_data(path, data):
    with open(path, "w") as filehandle:
        filehandle.write(serizalize_corpus_data(data))


def unigram_bigram_tagger(train_sentences):
    return UnigramTagger(
        train_sentences,
        backoff=BigramTagger(
            train_sentences,
            backoff=DefaultTagger("NN")
        )
    )


def unigram_trigram_tagger(train_sentences):
    return UnigramTagger(
        train_sentences,
        backoff=TrigramTagger(
            train_sentences,
            backoff=DefaultTagger("NN")
        )
    )


def unigram_bigram_trigram_tagger(train_sentences):
    return UnigramTagger(
        train_sentences,
        backoff=BigramTagger(
            train_sentences,
            backoff=TrigramTagger(
                train_sentences,
                backoff=DefaultTagger("NN")
            )
        )
    )

if __name__ == "__main__":
    for corpus_name in corpora:
        corpus_data = deserialize_corpus_data(open(corpora[corpus_name], "r").readlines())
        corpus_data = np.array(corpus_data, dtype=object)

        fold_counter = 1
        for train_index, test_index in KFold(n_splits=N_FOLD).split(corpus_data):
            # Cetak informasi data uji & data latih
            print("Fold ke-{}:".format(fold_counter))
            print("Data latih: Baris ke-{} sampai {}".format(train_index[0] + 1, train_index[-1] + 1))
            print("Data uji: Baris ke-{} sampai {}".format(test_index[0] + 1, test_index[-1] + 1))
            print("")

            train_data = corpus_data[train_index].tolist()
            test_data = corpus_data[test_index].tolist()


            save_corpus_data(get_training_data_path(corpus_name, fold_counter), train_data)
            save_corpus_data(get_test_data_path(corpus_name, fold_counter), test_data)
            

            unigram_bigram_tagger_model = unigram_bigram_tagger(train_data)
            unigram_trigram_tagger_model = unigram_trigram_tagger(train_data)
            unigram_bigram_trigram_tagger_model = unigram_bigram_trigram_tagger(train_data)


            pickle.dump(unigram_bigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_BIGRAM, fold_counter), "wb"))
            pickle.dump(unigram_trigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_TRIGRAM, fold_counter), "wb"))
            pickle.dump(unigram_bigram_trigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_BIGRAM_TRIGRAM, fold_counter), "wb"))

            fold_counter = fold_counter + 1
            pass

