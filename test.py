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

pos_types_map = {pos_type: index for index, pos_type in enumerate(pos_types)}

def report_confusion_matrix(y_pred: list, y_true: list, name: str, type: str, fold: int, labels: list) -> None:
    report_confusion_matrix = confusion_matrix(y_pred, y_true, labels=labels)
    report_confusion_matrix_df = pd.DataFrame(report_confusion_matrix)
    row_sums = report_confusion_matrix_df.T.sum(axis=1).to_list()
    
    text = f'''[{name}-{type}-fold-{fold_counter}]\n'''
    for row_pos_type in pos_types_map:
        row_index = pos_types_map[row_pos_type]
        text += f'''
Untuk POS dengan jenis '{row_pos_type}', terdapat {row_sums[row_index]} token yang tergolong \
ke dalam jenis tersebut.'''.strip() + ' '

        temp = []
        for col_pos_type in pos_types_map:
            col_index = pos_types_map[col_pos_type]

            temp.append("{count} diantaranya terklasifikasikan sebagai '{pos_type}'".format(
                count = report_confusion_matrix_df[row_index][col_index],
                pos_type = col_pos_type
            ).strip())
        text += ", ".join(temp[0:-1])
        text += ", dan " + temp[-1] + ".\n"

    with open("./reports/{}-{}-fold-{}.txt".format(name, type, fold), "w") as filehandle:
        filehandle.write(text)

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

            report_confusion_matrix(y_pred, y_true, corpus_name, model_type, fold_counter, labels=pos_types)
            report.to_excel("./reports/{}-{}-fold-{}.xlsx".format(corpus_name, model_type, fold_counter))
        pass
    pass
