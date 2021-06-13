from keras.models import Sequential
model = Sequential()
from keras.layers import Convolution2D
# Convolution
model.add(Convolution2D(
    filters = 64,
    kernel_size=(3,3),
    input_shape = (64, 64,3),
    activation='relu'
))
from keras.layers import MaxPooling2D
# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
from keras.layers import Flatten
# Flatten layer
model.add(Flatten())
from keras.layers import Dense
from keras.optimizers import Adam
# 1 NN Layer
model.add(Dense(
    units = 1550,
    activation = 'relu',
#    kernel_initializer='Zeros',
))
# 2 NN Layer
model.add(Dense(
    units = 500,
    activation = 'relu',
))
# 3 NN Layer
model.add(Dense(
    units = 200,
    activation = 'relu',
))
# 4 NN Layer
model.add(Dense(
    units = 70,
    activation = 'relu',
))
# 5 NN Layer
model.add(Dense(
    units = 1,
    activation = 'sigmoid',
))

# Optimizer & Loss function
model.compile(optimizer=Adam(learning_rate=0.00000001),loss='binary_crossentropy', metrics = ['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'ds/Training/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'ds/Training/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
)
model.save('ayush_recognition_final.h5')
import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['loss'])
