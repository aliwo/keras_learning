from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import optimizers


input = Input((6,))
x = Dense(64)(input)
x = Dropout(0.2)(x)
x = Dense(32)(x)
x = Dropout(0.2)(x)
x = Dense(16)(x)
output = Dense(2)(x)

model = Model(input, output)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
