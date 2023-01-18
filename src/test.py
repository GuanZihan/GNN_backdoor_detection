import numpy as np

dataset = np.load("../node_matrix_{}.npy".format("ENZYMES"), allow_pickle=True)
print(dataset)