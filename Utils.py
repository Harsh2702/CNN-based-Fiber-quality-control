# utils.py
import matplotlib.pyplot as plt

def plot_accuracy(train_acc, val_acc):
    plt.figure(figsize=(8,5))
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.legend(); plt.grid(); plt.show()


