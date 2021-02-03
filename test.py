from preprocess_and_train import corpora, N_FOLD, get_test_data_path, get_model_path, deserialize_corpus_data, model_types
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(30,20)})

labels = ['VBI', 'CDI', 'CON', '.', 'CDP', 'RP', 'IN', 'MD', 'DRB', 'PRL', 'NEG', 'TRB', 'DT', 'NN', 'UH', 'NNC', 'NNU', 'WPRB', ',', 'JJ', 'RB', 'WDT', 'PRP', 'NNG', 'VBT', 'NNP']

for corpus_name in corpora:
    for fold_counter in range(1, N_FOLD + 1):
        test_filepath = get_test_data_path(corpus_name, fold_counter)
        test_data = deserialize_corpus_data(
            open(test_filepath, "r").readlines())

        x_raw = [[part[0] for part in line] for line in test_data]
        x = [item for sublist in x_raw for item in sublist]

        for model_type in model_types:
            model = pickle.load(
                open(get_model_path(corpus_name, model_type, fold_counter), "rb")
            )

            y_pred_raw = model.tag_sents(x_raw)
            y_pred_raw = [[part[1] for part in line] for line in y_pred_raw]
            y_pred = [item for sublist in y_pred_raw for item in sublist]

            y_true_raw = [[part[1] for part in line] for line in test_data]
            y_true = [item for sublist in y_true_raw for item in sublist]

            report = pd.DataFrame(classification_report(
                y_pred,
                y_true,
                zero_division=0,
                output_dict=True
            ))

            cm = confusion_matrix(y_pred, y_true, labels=labels)

            heatmap = sns.heatmap(
                cm,
                annot=True,
                xticklabels=labels,
                yticklabels=labels,
                cmap="YlGnBu",
                linewidths=.3,
                linecolor="black",
                fmt=''
            )
            fig = heatmap.get_figure()
            fig.savefig("./reports/{}-{}-{}".format(
                corpus_name,
                model_type,
                fold_counter,
            ))

            plt.clf()

            report.to_excel("./reports/{}-{}-{}.xlsx".format(
                corpus_name,
                model_type,
                fold_counter
            ))
        pass
    pass
