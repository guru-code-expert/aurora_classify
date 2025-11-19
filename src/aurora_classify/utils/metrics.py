import matplotlib.pyplot as plt

def plot_knn_accuracy(k_range, accuracies, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o')
    plt.title("KNN Accuracy vs. Number of Neighbors")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()