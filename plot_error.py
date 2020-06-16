import matplotlib.pyplot as plt
import numpy as np
import sys

i = 1
while i < len(sys.argv) - 1:
    x, y = np.loadtxt(sys.argv[i], delimiter=' ', unpack=True)

    scaling = False
    if scaling:
        for j in range(1, len(y)):
            y[j] = y[j] / y[0]
        y[0] = 1.0;

    plt.scatter(x, y, s=3)
    plt.plot(x, y, '.r-')
    i = i + 1

plt.ylabel("Error")
plt.xlabel("Time")
# plt.savefig(sys.argv[2], bbox_inches='tight')
plt.show()
