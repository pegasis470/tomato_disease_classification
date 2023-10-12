import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import TomatoPreprocessing as pr
import matplotlib.pyplot as plt
import sys

folders=os.listdir('data')
def Model_small():
    tf.keras.backend.clear_session()
    inputs=inputs = tf.keras.Input(shape=(256, 256, 3))
    convo1= tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)
    maxpool1=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo1)
    convo2= tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(maxpool1)
    maxpool2=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo2)
    flatten=tf.keras.layers.Flatten()(maxpool2)
    #dropout1=tf.keras.layers.Dropout(0.5)(flatten)
    dence1=tf.keras.layers.Dense(100,activation='relu')(flatten)
    outputs=tf.keras.layers.Dense(10,activation='softmax')(dence1)
    #outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dence1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer ='adam' , loss ='categorical_crossentropy', metrics = ['accuracy'])
    return model
def model_Sq():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(250,250,3)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
#    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
#    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
#    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
#    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(70,activation='relu'))
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer ='adam' , loss = 'categorical_crossentropy', metrics = ['acc'])
    model.build(input_shape=(250,250))
    return model


if __name__=='__main__':
    #pr.AUG()
    #pr.Train_test_split('old')
    datagen = ImageDataGenerator(rescale=1./255)
    training_set = datagen.flow_from_directory('train',target_size=(250,250),batch_size = 10,classes=folders)
    validation_set = datagen.flow_from_directory('test',target_size=(250,250),batch_size = 10,classes=folders)
    model=model_Sq()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
    history=model.fit(training_set, validation_data = validation_set, epochs = 100 ,steps_per_epoch=250 ,callbacks=[callback])
    print('The final accuracy of the model against validtion set: ',history.history['val_acc'][-1]*100,'%')
    plot(history)
    for layer in model.layers:
        layer.trainable = False
    model.save(f'Tomato_Model.keras')
