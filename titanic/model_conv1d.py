from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.models import Model
from keras import optimizers



input = Input((6,))
x = Conv1D(64, kernel_size=6)(input)
x = MaxPooling1D()(x)
x = Conv1D(32, kernel_size=6)(x)
x = MaxPooling1D()(x)
x = Conv1D(16, kernel_size=3)(x)
x = GlobalMaxPool1D()(x)
output = Dense(2)(x)

model = Model(input, output)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
