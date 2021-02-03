#!/usr/bin/python

from nltk.corpus.reader.chasen import test
from nltk.tag.sequential import BigramTagger, DefaultTagger, TrigramTagger
from sklearn.model_selection import KFold
from nltk.tag import UnigramTagger
import numpy as np
from sklearn.metrics import classification_report
from pandas import DataFrame


corpus_filepath = "./corpora/corpus_1.txt"

corpus_data = []
with open(corpus_filepath, "r") as corpus_filehandle:
    for line in corpus_filehandle:
        corpus_data.append(line.strip())
        pass
    pass


def create_pair(pair_text, delimiter='/'):
    token, tag = pair_text.split(delimiter)
    return (token, tag)


def backoff_tagger(train_sentences, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sentences, backoff=backoff)
    return backoff


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


corpus_data = [line.split(' ') for line in corpus_data]
corpus_data = [[create_pair(part) for part in line] for line in corpus_data]
corpus_data = np.array(corpus_data)

fold_counter = 1
for train_index, test_index in KFold().split(corpus_data):
    # Cetak informasi data uji & data latih
    print("Fold ke-{}:".format(fold_counter))
    print(
        "Data latih: Baris ke-{} sampai {}".format(train_index[0] + 1, train_index[-1] + 1))
    print(
        "Data uji: Baris ke-{} sampai {}".format(test_index[0] + 1, test_index[-1] + 1))
    print("")

    train_data = corpus_data[train_index].tolist()
    test_data = corpus_data[test_index].tolist()

    unigram_bigram_tagger_model = unigram_bigram_tagger(train_data)
    unigram_trigram_tagger_model = unigram_trigram_tagger(train_data)
    unigram_bigram_trigram_tagger_model = unigram_bigram_trigram_tagger(
        train_data)

    x = [[part[0] for part in line] for line in test_data]
    y_true = [[part[1] for part in line] for line in test_data]
    flat_y_true = [item for sublist in y_true for item in sublist]

    y_pred_raw = unigram_bigram_tagger_model.tag_sents(x)
    y_pred = [[part[1] for part in line] for line in y_pred_raw]
    flat_y_pred = [item for sublist in y_pred for item in sublist]

    reports = {
        "unigram_bigram": DataFrame(classification_report(flat_y_true, flat_y_pred, zero_division=0, output_dict=True)),
        "unigram_trigram": DataFrame(classification_report(flat_y_true, flat_y_pred, zero_division=0, output_dict=True)),
        "unigram_bigram_trigram": DataFrame(classification_report(flat_y_true, flat_y_pred, zero_division=0, output_dict=True)),
    }

    for report_key in reports:
        reports[report_key].to_excel(
            "{}_fold_{}.xlsx".format(
                report_key,
                fold_counter,
            )
        )

    # pickle.dump(unigram_bigram_tagger_model, open("unigram_bigram_tagger.model", "wb"))
    # pickle.dump(unigram_trigram_tagger_model, open("unigram_trigram_tagger.model", "wb"))
    # pickle.dump(unigram_bigram_trigram_tagger_model, open("unigram_bigram_trigram_tagger.model", "wb"))

    fold_counter = fold_counter + 1
    pass
