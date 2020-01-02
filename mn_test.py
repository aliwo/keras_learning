from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

for X, Y in zip(X_train, Y_train):
    print(X, Y)
