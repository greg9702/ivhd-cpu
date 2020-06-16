import matplotlib.pyplot as plt
import numpy as np
import sys

x, y = np.loadtxt(sys.argv[1], delimiter=' ', unpack=True)

plt.scatter(x, y, s=3)
plt.plot(x, y, '.b-')

plt.ylabel("knn metric")
plt.xlabel("Iteration")
plt.title("Dataset " + sys.argv[2] + ", optimizer " + sys.argv[3] + ", k=" + sys.argv[4])
plt.savefig(sys.argv[5], bbox_inches='tight')
plt.show()
