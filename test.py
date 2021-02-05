from pandas.core.frame import DataFrame
from preprocess_and_train import corpora, N_FOLD, get_test_data_path, get_model_path, deserialize_corpus_data, model_types
from sklearn.metrics import classification_report
from pycm import ConfusionMatrix
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

sns.set(
    rc={'figure.figsize':(40,40)},
    font="Monospace",
)

def report_confusion_matrix(
    y_pred: list,
    y_true: list,
    corpus_name: str,
    algorithm_name: str,
    fold: int,
) -> None:
    confusion_matrix = ConfusionMatrix(
        actual_vector=y_true,
        predict_vector=y_pred,
    )

    confusion_matrix_positions = DataFrame(confusion_matrix.position())
    confusion_matrix_positions = confusion_matrix_positions.applymap(lambda positions: len(positions))

    report = DataFrame(classification_report(
        y_pred,
        y_true,
        zero_division=0,
        output_dict=True,
    ))

    report_text = ""
    for pos_class in confusion_matrix.classes:
        tp = confusion_matrix_positions[pos_class]["TP"]
        tn = confusion_matrix_positions[pos_class]["TN"]
        fp = confusion_matrix_positions[pos_class]["FP"]
        fn = confusion_matrix_positions[pos_class]["FN"]
        accuracy = tp + tn / (tp + tn + fp + fn)

        report_text += """
Untuk pengujian pada fold ke-{fold_counter} dari algoritma {algorithm_name}, kelas '{pos_class}' memiliki nilai true positive (tp) = {tp:{int_format}}, true negative (tn) = {tn:{int_format}}, false positive (fp) = {fp:{int_format}}, false negative (fn) = {fn:{int_format}}. Nilai recall = tp / tp + fn = {recall:{dec_format}}, nilai precision = tp / tp + fp = {precision:{dec_format}}, nilai f1-score = 2tp / (2tp + fp + fn) = {f1_score:{dec_format}}, accuracy = tp + tn / tp + tn + fp + fn = {accuracy:{dec_format}}.
""".format(
    algorithm_name=algorithm_name,
    pos_class=pos_class,
    tn=tn,
    fn=fn,
    tp=tp,
    fp=fp,
    fold_counter=fold_counter,
    precision=report[pos_class]["precision"],
    recall=report[pos_class]["recall"],
    f1_score=report[pos_class]["f1-score"],
    accuracy=accuracy,
    int_format="d",
    dec_format=".4f",
).strip()
    
    with open("./reports/{}-{}-fold-{}.txt".format(corpus_name, algorithm_name, fold_counter), "w") as report_filehandle:
        report_filehandle.write(report_text)

    pos_classes = confusion_matrix.classes
    confusion_matrix_df = DataFrame(confusion_matrix.to_array())
    row_sums = confusion_matrix_df.sum(axis=1).to_list()

    x_labels = ["{} ({})".format(label, row_sums[index]) for index, label in enumerate(pos_classes)]    

    temp_df = confusion_matrix_df.T

    shape = temp_df.shape
    annotations = [
        [f'''{pos_classes[col]}\n{pos_classes[row]}\n{temp_df[row][col]}''' 
            for col in range(0, shape[0])]
                for row in range(0, shape[1])
    ]

    confusion_matrix_heatmap = sns.heatmap(
        confusion_matrix_df.T,
        annot=annotations,
        xticklabels=x_labels,
        yticklabels=pos_classes,
        cmap="Greens",
        square=True,
        linewidths=.3,
        linecolor="black",
        fmt='',
        cbar=False,
    )

    fig = confusion_matrix_heatmap.get_figure()
    fig.savefig("./reports/{}-{}-fold-{}.svg".format(corpus_name, algorithm_name, fold,))
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

            report_confusion_matrix(y_pred, y_true, corpus_name, model_type, fold_counter)
            report.to_excel("./reports/{}-{}-fold-{}.xlsx".format(corpus_name, model_type, fold_counter))
        pass
    pass
