#!/usr/bin/python

from matplotlib.pyplot import annotate
from nltk.corpus.reader.chasen import test
from nltk.tag.sequential import BigramTagger, DefaultTagger, TrigramTagger
from pandas.core.common import flatten
import seaborn as sns
from seaborn.palettes import color_palette
from sklearn.model_selection import KFold
from nltk.tag import UnigramTagger
import numpy as np
from pandas import DataFrame
import pickle

sns.set(
    rc={'figure.figsize':(20,17)},
    font="Monospace",
)

# Jumlah fold
N_FOLD = 5

# Nama berkas untuk masing-masing metode
UNIGRAM_BIGRAM = "unigram_bigram"
UNIGRAM_TRIGRAM = "unigram_trigram"
UNIGRAM_BIGRAM_TRIGRAM = "unigram_bigram_trigram"

model_types = [
    UNIGRAM_BIGRAM,
    UNIGRAM_TRIGRAM,
    UNIGRAM_BIGRAM_TRIGRAM
]

corpora = {
    "corpus_1500": "./corpora/korpus_tagging_1500.crp",
    "corpus_1": "./corpora/corpus_1.txt",
    "corpus_wicaksono": "./corpora/corpus_wicaksono.crp",
}

# Format nama berkas model hasil training
def get_model_path(name, type, fold):
    return "./models/{}-{}-{}.model".format(name, type, fold)

# Format nama berkas data training
def get_training_data_path(name, fold):
    return "./training_data/{}-{}.txt".format(name, fold)

# Format nama berkas data uji
def get_test_data_path(name, fold):
    return "./test_data/{}-{}.txt".format(name, fold)

# Proses data korpus sebelum disave
def serizalize_corpus_data(training_data):
    return "\n".join([
        " ".join(["/".join(part) for part in line])
            for line in training_data
    ])

# Memecah teks dengan format KATA/POS menjadi [KATA, POS]
def pair_to_tuple(pair_text):
    token, pos = pair_text.split("/")
    return (token, pos)

# Load data korpus
def deserialize_corpus_data(lines):
    try:
        results = []
        for index, line in enumerate(lines):
            split_line = line.strip().split(" ")
            split_parts = [pair_to_tuple(part) for part in split_line]
            results.append(split_parts)
        return results
    except ValueError as value_error:
        print("Gagal memroses baris {}: {}".format(
            index + 1,
            line,
        ))
        raise value_error

# Save data korpus ke berkas
def save_corpus_data(path, data):
    with open(path, "w") as filehandle:
        filehandle.write(serizalize_corpus_data(data))

# Logika training Unigram-Bigram
def unigram_bigram_tagger(train_sentences):
    return UnigramTagger(
        train_sentences,
        backoff=BigramTagger(
            train_sentences,
            backoff=DefaultTagger("NN")
        )
    )

# Logika training Unigram-Trigram
def unigram_trigram_tagger(train_sentences):
    return UnigramTagger(
        train_sentences,
        backoff=TrigramTagger(
            train_sentences,
            backoff=DefaultTagger("NN")
        )
    )

# Logika training Unigram-Bigram-Trigram
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

# Logika pembuatan laporan karakterisik korpus
def report_corpus(corpus: list, name: str):
    pos_class_df = DataFrame([
        pos for pos in
        flatten([[part[1] for part in line] for line in corpus_data])
    ], columns=["POS"])

    grouped_by_pos_df = pos_class_df.groupby(
        "POS"
    ).size(
    ).to_frame(
        name="frequency"
    ).reset_index(
    ).sort_values(
        by="frequency"
    )

    grouped_by_pos_df.to_excel("./reports/{}-class-stats.xlsx".format(
        name
    ))

    grouped_by_pos_df["POS"] = grouped_by_pos_df.apply(
        lambda row: "{} ({})".format(row["POS"],
        row["frequency"]), axis=1
    )

    sns.barplot(
        data=grouped_by_pos_df,
        x="frequency",
        y="POS",
        orient="h",
        saturation=1,
        palette="tab10",
    ).get_figure(
    ).savefig(
        "./reports/{}-corpus-barplot.svg".format(name)
    )
    
    pass

if __name__ == "__main__":
    # Load data korpus satu per sau
    for corpus_name in corpora:
        corpus_data = deserialize_corpus_data(open(corpora[corpus_name], "r").readlines())
        report_corpus(corpus_data, corpus_name)

        corpus_data = np.array(corpus_data, dtype=object)

        fold_counter = 1

        print("Memroses korpus {}...".format(corpus_name))

        # Belah korpus menjadi sekian fold
        for train_index, test_index in KFold(n_splits=N_FOLD).split(corpus_data):
            # Cetak informasi data uji & data latih
            print("Fold ke-{}:".format(fold_counter))
            print("Data latih: Baris ke-{} sampai {}".format(train_index[0] + 1, train_index[-1] + 1))
            print("Data uji: Baris ke-{} sampai {}".format(test_index[0] + 1, test_index[-1] + 1))
            print("")

            train_data = corpus_data[train_index].tolist()
            test_data = corpus_data[test_index].tolist()

            # Save data training & uji
            save_corpus_data(get_training_data_path(corpus_name, fold_counter), train_data)
            save_corpus_data(get_test_data_path(corpus_name, fold_counter), test_data)
            
            # Pembuatan model
            unigram_bigram_tagger_model = unigram_bigram_tagger(train_data)
            unigram_trigram_tagger_model = unigram_trigram_tagger(train_data)
            unigram_bigram_trigram_tagger_model = unigram_bigram_trigram_tagger(train_data)

            # Save model
            pickle.dump(unigram_bigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_BIGRAM, fold_counter), "wb"))
            pickle.dump(unigram_trigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_TRIGRAM, fold_counter), "wb"))
            pickle.dump(unigram_bigram_trigram_tagger_model, open(get_model_path(corpus_name, UNIGRAM_BIGRAM_TRIGRAM, fold_counter), "wb"))

            fold_counter = fold_counter + 1
            pass

