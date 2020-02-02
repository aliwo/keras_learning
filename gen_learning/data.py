from keras.utils import Sequence
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

class MnistGenerator(Sequence):

    def __init__(self, batch_size=128):
        (train_images, train_labels), _ = mnist.load_data()
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255

        self.train_images = train_images
        self.train_labels = to_categorical(train_labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.train_images) / self.batch_size))

    def __getitem__(self, idx):
        return (self.train_images[idx * self.batch_size : (idx + 1) * self.batch_size],
                self.train_labels[idx * self.batch_size : (idx + 1) * self.batch_size])


_, (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)





