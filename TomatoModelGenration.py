import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import TomatoPreprocessing as pr
import matplotlib.pyplot as plt
import sys
import pdb
folders=os.listdir('data')
def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy')
    plt.legend([ 'train','test','train_loss','test_loss'], loc='upper left')
    plt.savefig("model_performance.png")

def Model_small():
    tf.keras.backend.clear_session()
    inputs=inputs = tf.keras.Input(shape=(150, 150, 3))
    convo1= tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)
    maxpool1=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo1)
    convo2= tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(maxpool1)
    maxpool2=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo2)
    flatten=tf.keras.layers.Flatten()(maxpool2)
    #dropout1=tf.keras.layers.Dropout(0.5)(flatten)
    dence1=tf.keras.layers.Dense(100,activation='relu')(flatten)
    #dence2=tf.keras.layers.Dense(100,activation='relu')(dence1)
    #dence3=tf.keras.layers.Dense(100,activation='relu')(dence2)
    dence4=tf.keras.layers.Dense(50,activation='relu')(dence1)
    dence5=tf.keras.layers.Dense(25,activation='relu')(dence4)
    outputs=tf.keras.layers.Dense(10,activation='softmax')(dence5)
    #outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dence1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer ='adam' , loss ='categorical_crossentropy', metrics = ['acc'])
    return model
def model_Sq():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(150,150,3)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
#    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
#    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Flatten())
#    model.add(tf.keras.layers.Dense(100,bias_initializer='zeros',activation='relu'))
    model.add(tf.keras.layers.Dense(70,bias_initializer='zeros',activation='relu'))
    model.add(tf.keras.layers.Dense(20,bias_initializer='zeros',activation='relu'))
    model.add(tf.keras.layers.Dense(10,bias_initializer='zeros',activation='relu'))
    model.add(tf.keras.layers.Dense(10,bias_initializer='zeros',activation='softmax'))
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer = opt , loss = 'categorical_crossentropy', metrics = ['acc',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.build(input_shape=(50,50))
    return model


if __name__=='__main__':
   # pr.AUG()
    pr.Train_test_split('new')
    datagen = ImageDataGenerator(rescale=1./255)
    training_set = datagen.flow_from_directory('train',target_size=(150,150),batch_size = 10,classes=folders)
    validation_set = datagen.flow_from_directory('test',target_size=(150,150),batch_size = 10,classes=folders)
    model=model_Sq()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3,restore_best_weights = True)
    tf.keras.utils.plot_model(model,to_file='model.png')
    history=model.fit(training_set, validation_data = validation_set, epochs = 20  ,callbacks=[callback])
    print('The final accuracy of the model against validtion set: ',history.history['val_acc'][-1]*100,'%')
    plot(history)
    for layer in model.layers:
        layer.trainable = False
    model.save(f'Tomato_Model.keras')
