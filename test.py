from preprocess_and_train import corpora, N_FOLD, get_test_data_path, get_model_path, deserialize_corpus_data, model_types
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(40,20)})

pos_types = [
    'VBI', 'CDI', 'CON', '.',
    'CDP', 'RP', 'IN', 'MD',
    'DRB', 'PRL', 'NEG', 'TRB',
    'DT', 'NN', 'UH', 'NNC',
    'NNU', 'WPRB', ',', 'JJ',
    'RB', 'WDT', 'PRP', 'NNG',
    'VBT', 'NNP'
]

def plot_report_confusion_matrix(y_pred: list, y_true: list, name: str, type: str, fold: int, labels: list) -> None:
    report_confusion_matrix = confusion_matrix(y_pred, y_true, labels=labels)
    temp_df = pd.DataFrame(report_confusion_matrix)
    row_sums = temp_df.T.sum(axis=1).to_list()
    x_labels = ["{} ({})".format(label, row_sums[index]) for index, label in enumerate(labels)]
    
    confusion_matrix_heatmap = sns.heatmap(
        report_confusion_matrix,
        annot=True,
        xticklabels=x_labels,
        yticklabels=labels,
        cmap="OrRd",
        linewidths=.3,
        linecolor="black",
        fmt=''
    )

    fig = confusion_matrix_heatmap.get_figure()
    fig.savefig("./reports/{}-{}-fold-{}.svg".format(name, type, fold,))
    plt.clf()

for corpus_name in corpora:
    for fold_counter in range(1, N_FOLD + 1):
        test_filepath = get_test_data_path(corpus_name, fold_counter)
        test_data = deserialize_corpus_data(
            open(test_filepath, "r").readlines())

        x_raw = [[part[0] for part in line] for line in test_data]
        x = [item for sublist in x_raw for item in sublist]

        for model_type in model_types:
            print("Memroses corpus {}, fold {}, metode {}.".format(corpus_name, fold_counter, model_type))

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
                output_dict=True,
            ))

            plot_report_confusion_matrix(y_pred, y_true, corpus_name, model_type, fold_counter, labels=pos_types)
            report.to_excel("./reports/{}-{}-fold-{}.xlsx".format(corpus_name, model_type, fold_counter))
        pass
    pass
