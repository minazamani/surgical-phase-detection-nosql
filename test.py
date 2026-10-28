import numpy as np

folder = "/home/mina/PycharmProjects/PythonProject2/processed/U57L_1_raw"

x = np.load(folder + "/features.npy")
y = np.load(folder + "/phase_labels_aligned.npy")

print("features:", x.shape, x.dtype)
print("labels:", y.shape, y.dtype)

print("unique labels:", np.unique(y))
print("ignored samples:", np.sum(y == -100))
