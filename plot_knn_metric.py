import matplotlib.pyplot as plt
import numpy as np

label = ['SGD', 'Momentum', 'Nesterov', 'Adadelta', 'Adam']
k_5 = [0.6154485714285715, 0.6192028571428572, 0.6189028571428571, 0.4852457142857143, 0.6696571428571428]
k_15 = [0.615792380952381, 0.6194438095238095, 0.6186504761904762, 0.48527619047619047, 0.6684457142857143]
k_50 = [0.6146031428571429, 0.61845, 0.617798, 0.484606, 0.6671534285714286]

index = np.arange(len(label))
plt.bar(index, k_50)
plt.xlabel('Algorithm')
plt.ylabel('knn metric')
plt.xticks(index, label, rotation=30)
ax = plt.gca()
for index, value in enumerate(k_50):
    ax.text(index - 0.4, value / 2, str(round(value, 6)), color='black', fontweight='bold')
plt.title('knn metric for different optimization algorithms, k=50')
# plt.savefig("knn_metric_k_50", bbox_inches='tight')
plt.show()
