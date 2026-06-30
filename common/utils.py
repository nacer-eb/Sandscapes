import numpy as np

# For utils
# Utils only for printing array size inhuman readable format
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_data_partition(data, data_labels, selected_labels, N_per_class=100, replace=True, verbose=False, seed=42):
    rng = np.random.default_rng(seed)

    N_classes = np.max(data_labels)+1
    indices = np.zeros((len(selected_labels), N_per_class))


    for i in range(len(selected_labels)):
        if verbose:
            print(selected_labels[i], "Load", N_per_class, "cells from", np.sum(data_labels==selected_labels[i]))
        if replace==False and N_per_class>np.sum(data_labels==selected_labels[i]):
            print("Error too few", selected_labels[i], "to set replace to False")

        indices[i] = rng.choice(np.argwhere(data_labels==selected_labels[i]).ravel(), N_per_class, replace=replace)

    indices = np.asarray(indices.ravel(), dtype=int)

    data_partionned = data[indices]
    data_labels_partionned = data_labels[indices]

    return data_partionned, data_labels_partionned
