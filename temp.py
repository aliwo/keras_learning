import matplotlib.pyplot as plt
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

i = 0
for X in X_train:
    plt.figure(i)
    plt.imshow(X, cmap='gray')
    i += 1
    if i % 4 == 0:
        break

plt.show()



