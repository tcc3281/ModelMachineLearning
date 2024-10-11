import pandas as pd

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    data.to_csv(file_path, index=False)

def split_data(data, ratio):
    # ramdom sampling
    train_data = data.sample(frac=ratio)
    test_data = data.drop(train_data.index)
    return train_data, test_data

def equal_discrate_bins(data, bins, target=None):
    data = data.copy()
    for column in data.columns:
        if column == target:
            continue
        data[column] = pd.cut(data[column], bins, labels=False)
    return data

def k_fold_cross_validation(data, k):
    k_folds=[]
    shuffled_df = data.sample(frac=1).reset_index(drop=True)
    fold_size = int(len(data)/k)
    for i in range(k):
        start = i*fold_size
        end = (i+1)*fold_size
        k_folds.append(shuffled_df[start:end])
    return k_folds
