from gen_learning.data import MnistGenerator
from gen_learning.data import test_images, test_labels
from gen_learning.model import network

network.fit_generator(MnistGenerator(), epochs=5)
network.save('mnist.h5')

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
