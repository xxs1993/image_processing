import cv2
import os;

import keras
import numpy as np
from keras import Sequential, Input, Model
from keras.engine import InputLayer
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras_preprocessing.image import ImageDataGenerator

base_path = "drive/My Drive/Colab Notebooks"
train_path = base_path+"/train"
test_path = base_path+"/test"
model_name = "inceptionV3_update_model.h5"
cv2.imread(model_name, cv2.IMREAD_GRAYSCALE)

def train_data_set(path):
    data_set = []
    for dir in os.listdir(path):
        new_path = os.path.join(path,dir)
        for file in os.listdir(new_path):
            file_path = os.path.join(new_path,file)
            img = cv2.imread(file_path)
            img = cv2.resize(img,(299,299))
            data_set.append([np.array(img),dir])
    return data_set
train_data = train_data_set()
test_data = train_data_set()

train_data_img = np.array([i[0] for i in train_data]).reshape(-1,299,299,3).astype("float32")
train_data_label = np.array([i[1] for i in train_data]).reshape(-1,1)

test_data_img = np.array([i[0] for i in test_data]).reshape(-1,299,299,3).astype("float32")
test_data_label = np.array([i[1] for i in test_data]).reshape(-1,1)

train_data_label = keras.utils.to_categorical(train_data_label, num_classes=4)
test_data_label = keras.utils.to_categorical(test_data_label, num_classes=4)
# input_tensor = Input(input_shape=(299,299,3))
input = Input(shape=(299,299,3),name = 'image_input')
vgg16_model = keras.applications.inception_resnet_v2
# vgg16_model = keras.applications.inception_v3.InceptionV3(include_top = False, weights="imagenet")
# vgg16_model.summary()

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    Conv2D()
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# model = add_new_last_layer(vgg16_model,4)

# setup_to_transfer_learn(model, vgg16_model)


def setup_to_finetune(model):
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# setup_to_finetune(model)

from keras.models import load_model
if os.path.exists(model_path):
    vgg16_model = load_model(model_path)
else:
    vgg16_model = keras.applications.resnet50.ResNet50(include_top = False, weights="imagenet")
    vgg16_model.summary()
    model = add_new_last_layer(vgg16_model, 4)
    setup_to_transfer_learn(model, vgg16_model)
    setup_to_finetune(model)
print(model.evaluate(train_data_img, train_data_label, batch_size=300, verbose=0))

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=120,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format="channels_last",
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.2)
datagen.fit(train_data_img)
model.fit_generator(datagen.flow(train_data_img, train_data_label,
                                 batch_size=10),
                    steps_per_epoch=300,
                    validation_data=(test_data_img,test_data_label),
                    shuffle=True,
                    workers=4)

##model.fit(train_data_img,train_data_label,batch_size=10,epochs=300,shuffle=True,validation_data=(test_data_img,test_data_label))
model.save(base_path+model_name)