from pandas.core.frame import DataFrame
from preprocess_and_train import corpora, N_FOLD, get_test_data_path, get_model_path, deserialize_corpus_data, model_types
from sklearn.metrics import classification_report
from pycm import ConfusionMatrix
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Setting plot confusion matrix (ukuran font, gambar)
sns.set(
    font_scale=0.65,
    rc={'figure.figsize': (22, 20)},
    font="Monospace",
)

# Prosedur pembuatan laporan
def generate_and_save_report(
    y_pred: list,
    y_true: list,
    corpus_name: str,
    algorithm_name: str,
    fold: int,
) -> None:
    report = DataFrame(classification_report(
        y_pred,
        y_true,
        zero_division=0,
        output_dict=True,
    ))

    # Save laporan ke Excel
    report.to_excel(
        "./reports/{}-{}-fold-{}.xlsx".format(corpus_name, algorithm_name, fold))

    # Buat plot Confusion Matrix untuk masing-masing kelas pengelompokan
    confusion_matrix = ConfusionMatrix(
        actual_vector=y_true,
        predict_vector=y_pred,
    )

    confusion_matrix_positions = DataFrame(confusion_matrix.position())
    confusion_matrix_positions = confusion_matrix_positions.applymap(
        lambda positions: len(positions))

    pos_classes = confusion_matrix.classes
    pos_classes.sort()

    report_text = ""
    for pos_class in pos_classes:
        tp = confusion_matrix_positions[pos_class]["TP"]
        tn = confusion_matrix_positions[pos_class]["TN"]
        fp = confusion_matrix_positions[pos_class]["FP"]
        fn = confusion_matrix_positions[pos_class]["FN"]
        accuracy = tp + tn / (tp + tn + fp + fn)

        report_text += get_pos_class_report_text(
            algorithm_name, report, pos_class, tp, tn, fp, fn, accuracy)

    # Simpan teks laporan
    with open("./reports/{}-{}-fold-{}.txt".format(corpus_name, algorithm_name, fold_counter), "w") as report_filehandle:
        report_filehandle.write(report_text)

    confusion_matrix_df = DataFrame(confusion_matrix.to_array())

    row_sums = confusion_matrix_df.sum().to_list()

    y_labels = pos_classes
    x_labels = [
        "{} ({})".format(label, row_sums[index]) for index, label in enumerate(pos_classes)
    ]

    temp_df = confusion_matrix_df.T

    shape = temp_df.shape
    annotations = [
        [f'''{pos_classes[col]}\n{pos_classes[row]}\n{temp_df[row][col]}'''
            for col in range(0, shape[0])]
        for row in range(0, shape[1])
    ]

    confusion_matrix_heatmap = sns.heatmap(
        confusion_matrix_df,
        annot=annotations,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="Greens",
        linewidths=0.1,
        linecolor="black",
        fmt='',
        cbar=False,
    )

    fig = confusion_matrix_heatmap.get_figure()
    fig.savefig(
        "./reports/{}-{}-fold-{}.svg".format(corpus_name,
                                             algorithm_name, fold,),
        bbox_inches='tight'
    )
    plt.clf()
    pass


def get_pos_class_report_text(algorithm_name, report, pos_class, tp, tn, fp, fn, accuracy):
    return """ Untuk pengujian pada fold ke-{fold_counter} dari algoritma {algorithm_name}, kelas '{pos_class}' memiliki nilai true positive (tp) = {tp:{int_format}}, true negative (tn) = {tn:{int_format}}, false positive (fp) = {fp:{int_format}}, false negative (fn) = {fn:{int_format}}. Nilai recall = tp / tp + fn = {recall:{dec_format}}, nilai precision = tp / tp + fp = {precision:{dec_format}}, nilai f1-score = 2tp / (2tp + fp + fn) = {f1_score:{dec_format}}, accuracy = tp + tn / tp + tn + fp + fn = {accuracy:{dec_format}}. """.format(algorithm_name=algorithm_name, pos_class=pos_class, tn=tn, fn=fn, tp=tp, fp=fp, fold_counter=fold_counter, precision=report[pos_class]["precision"], recall=report[pos_class]["recall"], f1_score=report[pos_class]["f1-score"], accuracy=accuracy, int_format="d", dec_format=".4f", ).strip()

# Prosedur pelaporan untuk setiap korpus & setiap metode
for corpus_name in corpora:
    for fold_counter in range(1, N_FOLD + 1):
        test_filepath = get_test_data_path(corpus_name, fold_counter)
        # Load data uji
        test_data = deserialize_corpus_data(
            open(test_filepath, "r").readlines())

        x_raw = [[part[0] for part in line] for line in test_data]
        x = [item for sublist in x_raw for item in sublist]

        # Load masing-masing model dari metode yang telah dilatih
        for model_type in model_types:
            print("Memroses corpus {}, fold {}, metode {}.".format(
                corpus_name, fold_counter, model_type))

            model = pickle.load(
                open(get_model_path(corpus_name, model_type, fold_counter), "rb")
            )

            # Lakukan prediksi
            y_pred_raw = model.tag_sents(x_raw)
            y_pred_raw = [[part[1] for part in line] for line in y_pred_raw]
            y_pred = [item for sublist in y_pred_raw for item in sublist]

            y_true_raw = [[part[1] for part in line] for line in test_data]
            y_true = [item for sublist in y_true_raw for item in sublist]

            # Lakukan prosedur pelaporan
            generate_and_save_report(
                y_pred, y_true, corpus_name, model_type, fold_counter)
        pass
    pass
