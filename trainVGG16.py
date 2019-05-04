# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:10:30 2019

@author: Ajaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 04:21:07 2019

@author: Ajaz
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import os
from keras import models
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
from matplotlib import pyplot
with tf.device('/gpu:0'):
    baseDir= 'D:/Books/data/resized_224/'
    train_dir=os.path.join(baseDir,'train')
    validation_dir=os.path.join(baseDir,'validation')
    test_dir=os.path.join(baseDir,'test')
    
    
    def vgg_model():
        #create model
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return conv_base
    
    #build the model
    conv_base=vgg_model()
    sgd=SGD(lr=0.001, momentum=0.9,nesterov=True)
    #rms=RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=None)
    #new_opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    print('Model Summary',conv_base.summary())
    #data Augmentation
    #train_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=10,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    test_datagen=ImageDataGenerator(rescale=1./255)
    validation_datagen=ImageDataGenerator(rescale=1./255)
    train_generator=train_datagen.flow_from_directory(train_dir,target_size=(224,224), batch_size=16,classes=['Cardiomegaly','No Finding'],class_mode='binary')
    validation_generator=validation_datagen.flow_from_directory(train_dir,target_size=(224,224), batch_size=16,classes=['Cardiomegaly','No Finding'], class_mode='binary')
    
    model=models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    #model.add(Dense(32,activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(1,activation='sigmoid'))
    #conv_base.trainable=False
    model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy',optimizer=new_opt, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy',optimizer=rms, metrics=['accuracy'])
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=42, validation_data=validation_generator, validation_steps=24)
        #accuracy
    pyplot.plot(history.history['acc'],label='train')
    pyplot.plot(history.history['val_acc'],label='validation')
    pyplot.title('model Accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train','validation'],loc='upper left')
    pyplot.show()
    
    # summarize history for loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper left')
    pyplot.show()
    test_generator=test_datagen.flow_from_directory(
                test_dir, target_size=(224,224),batch_size=20,classes=['Cardiomegaly','No Finding'], class_mode='binary')
    
    # finally evaluate this model on the test data
    results = model.evaluate_generator(
                test_generator,steps=10)
    
    print('Final test accuracy:', (results[1]*100.0))    
