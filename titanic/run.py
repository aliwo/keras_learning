from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from titanic.model_conv1d import model
from titanic.prepare import load


(train_data, train_label), (val_data, val_label) = load()
history = model.fit(train_data, to_categorical(train_label), validation_data=(val_data, to_categorical(val_label)),
                    callbacks=[TensorBoard(log_dir='my_log_dir', histogram_freq=1)], epochs=30)

model.save('titanic.h5')
