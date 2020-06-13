import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=4)


x, y, labels = np.loadtxt(sys.argv[1], delimiter=' ', unpack=True)

unique = list(set(labels))
plt.figure(figsize=(16, 9))
colors = cm.jet(np.linspace(0, 1, len(unique)))

for i, label in enumerate(unique):
    xi = [x[j] for j in range(len(x)) if labels[j] == label]
    yi = [y[j] for j in range(len(x)) if labels[j] == label]
    plt.scatter(xi, yi, c=[colors[i]], label=str(label), alpha=0.7)
legend_without_duplicate_labels(plt.gca())

plt.show()
