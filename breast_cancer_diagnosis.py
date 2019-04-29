# Dataset from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
import numpy as np
import mlp
import csv

def csv_to_array(file_name):
    np_a = None
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            class_label = 0
            if row[0].strip()=='M':
                class_label = 1
            num_arr = [float(x) for x in row[1:]] + [class_label]

            if np_a is None:
                np_a = np.array(num_arr)
            else:
                np_a = np.c_[np_a, num_arr]

    np_a = np.transpose(np_a)

    return np_a


if __name__ == "__main__":
    bc_data = csv_to_array("wdbc.csv")
    n_rows = np.shape(bc_data)[0]
    n_cols = np.shape(bc_data)[1]
    print("dataset contains",n_rows,"rows and", n_cols,"columns")

    # keep target attribute 0 or 1
    target_a = bc_data[:,n_cols-1:n_cols ]
    # normalize data in each column by subtracting mean and dividing by variance
    bc_data = bc_data[:, 0:n_cols-1]
    bc_data = (bc_data - bc_data.mean(axis=0)) / bc_data.var(axis=0)

    bc_data = np.c_[bc_data,target_a]
    # print(bc_data)
    p = mlp.mlp(bc_data[:, 0:n_cols-1], bc_data[:, n_cols-1:n_cols], 2, outtype="linear")
    p.mlptrain(bc_data[:, 0:n_cols-1], bc_data[:, n_cols-1:n_cols], 0.0005, 1000001)
    p.confmat(bc_data[:, 0:n_cols-1], bc_data[:, n_cols-1:n_cols])