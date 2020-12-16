import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from dao import MovieLens100kDao as Dao
from pu_dataset import PUDataset
from deepfm import DeepFM as BaseModel


def calibrate_pu(y_pred, y_val_pos_pred):
    val_pos_average = np.mean(y_val_pos_pred)
    return y_pred / val_pos_average


def normalized_entropy(y_true, y_proba):
    p = np.mean(y_true)
    frac = - np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
    deno = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return frac / deno

def binarize(y_proba, threshold=0.5):
    return np.array([
            1.0 if p > threshold else 0
            for p in y_proba
        ])

def main():
    # build ordinal dataset
    dao = Dao()
    dao.build()
    X_train, X_test, y_train, y_test = dao.fetch_dataset()
    features_info = dao.features_info

    # pu learning dataset instance
    dataset = PUDataset(y_train, X_train)
    f1_scores = []
    pu_f1_scores = []
    labeled_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for labeled_rate in labeled_rates:
        # fetch ordinal pu dataset
        y_train, X_train = dataset.fetch_pu_dataset(labeled_rate, val_split=False)
        # fetch pu dataset with val
        y_train_pu, X_train_pu, X_val_pos = dataset.fetch_pu_dataset(labeled_rate, val_split=True)

        # train ordinal model
        auc = tf.keras.metrics.AUC()
        optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.1)
        model = BaseModel(features_info)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[auc])
        model.fit(x=X_train, y=y_train, epochs=100, batch_size=100000)

        # train pu model
        auc = tf.keras.metrics.AUC()
        pu_optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.1)
        pu_model = BaseModel(features_info)
        pu_model.compile(optimizer=pu_optimizer, loss="binary_crossentropy", metrics=[auc])
        pu_model.fit(x=X_train_pu, y=y_train_pu, epochs=100, batch_size=100000)

        print(labeled_rate)
        # evaluate ordinal model
        y_pred = np.ravel(model.predict(X_test))
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, binarize(y_pred))
        f1_scores.append(f1_score[1])
        print("ordinal model")
        print("F1 score: {}".format(f1_score[1]))
        print("Precision: {}".format(precision[1]))
        print("Recall: {}".format(recall[1]))

        # evaluate pu model
        y_pred_pu = np.ravel(pu_model.predict(X_test))
        y_val_pos_pred = np.ravel(pu_model.predict(X_val_pos))
        y_pred_pu = calibrate_pu(y_pred_pu, y_val_pos_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, binarize(y_pred_pu))
        pu_f1_scores.append(f1_score[1])
        print("pu model")
        print("F1 score: {}".format(f1_score[1]))
        print("Precision: {}".format(precision[1]))
        print("Recall: {}".format(recall[1]))
    
    # plot
    fig = plt.figure()
    plt.title("DeepFM with/without PU learning")
    plt.plot(labeled_rates, pu_f1_scores, label='PU Adapted DeepFM')
    plt.plot(labeled_rates, f1_scores, label='DeepFM')
    plt.xlabel('labeled sample rate of all positive samples')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.gca().invert_xaxis()
    fig.savefig("result.png")

if __name__ == "__main__":
    main()