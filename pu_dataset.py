import numpy as np

class PUDataset:
    def __init__(self, y_train, X_train, val_split_rate=0.2):
        self.label_pos = 1
        self.label_neg = 0
        self.label_unlabeled = 0
        self.val_split_rate = val_split_rate
        self.val_indices = None
        self.train_indices = None
        self.set_split_indices(y_train)
        self.y_train = y_train
        self.X_train = X_train
        print(X_train)

    def set_split_indices(self, y_train):
        if self.val_split_rate > 0:
            split_indices = np.random.permutation(len(y_train))
            print(split_indices)
            val_num = int(np.floor(len(y_train) * self.val_split_rate))
            self.val_indices = split_indices[:val_num]
            print(self.val_indices)
            self.train_indices = split_indices[val_num:]

    def fetch_pu_dataset(self, labeled_rate, val_split):
        y_train_pu, _ = self.convert_to_pu(labeled_rate)
        if val_split:
            X_val = self.split_X(self.X_train, self.val_indices)
            X_train = self.split_X(self.X_train, self.train_indices)
            y_val = y_train_pu[self.val_indices]
            y_train = y_train_pu[self.train_indices]

            val_pos_indices = np.where(y_val == self.label_pos)[0]
            X_val_pos = self.split_X(X_val, val_pos_indices)
            
            return y_train, X_train, X_val_pos
        else:
            return y_train_pu, self.X_train

    def split_X(self, X, split_indices):
        X_split = []
        for values in X:
            X_split.append(values[split_indices])
        return X_split

    def convert_to_pu(self, labeled_rate):
        """
        positive unlabeledに変換
        """
        y = np.copy(self.y_train)

        # 事前確率計算
        n_pos = (y == self.label_pos).sum()
        n_neg = (y == self.label_neg).sum()
        n_labeled_pos = int(np.floor(n_pos * labeled_rate))
        n_unlabeled_pos = n_pos - n_labeled_pos
        n_unlabeled = n_neg + n_labeled_pos
        prior = float(n_unlabeled_pos) / float(n_unlabeled)

        # positiveのうちunlabeledにするものを置き換え
        pos_indices = np.where(y == self.label_pos)[0]
        unlabeled_pos_indices = np.random.choice(pos_indices, n_unlabeled_pos, replace=False)
        np.put(y, unlabeled_pos_indices, self.label_neg)

        # label_negをlabel_unlabeledに変換
        y = np.where(y == self.label_neg, self.label_unlabeled, y)

        return y, prior