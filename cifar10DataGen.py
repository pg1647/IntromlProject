# -*- coding: utf-8 -*-
"""
Created on Sat May 12 02:19:58 2018

@author: Prachi Gupta
"""

import numpy as np
import keras
from keras.datasets import cifar10
from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model #save and load models
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import keras.backend as K
K.clear_session()

# The data, split between train and test sets
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#combining all the available data to use it in a way we want
x = np.vstack((X_train,X_test))
y = np.vstack((Y_train,Y_test))
print('x shape:', x.shape)
print('y shape:', y.shape)

# normalize inputs from 0-255 to 0.0-1.0
x = x.astype('float32')
x = x/255
num_classes = 10
y1 = keras.utils.to_categorical(y, num_classes)
print('y1 shape', y1.shape)
print('Number of classes:', y1.shape[1])

#Model parameters for target and shadow models
batch_size = 32 #upto us
epochs = 1
lrate = 0.001
decay = 1e-7 #find out what this decay parameter does
kernel_size = (5,5) #upto us
kernel_size2 = (3,3) #upto us
nout1 = 32 #upto us
nout2 = 32 #upto us
ndense = 128
#initializer in each layer - upto us

data_size = [10000] #[2500,5000,10000,15000]
target_rep = np.zeros((len(data_size),x.shape[0]))
ns = 1 #number of shadow models for one data_size

for i,ds in enumerate(data_size): 
    sh = np.arange(x.shape[0])
    np.random.shuffle(sh)
    target_rep[i,:] = sh
    xtr_target = x[sh[:ds]]
    ytr_target = y1[sh[:ds]]
    xts_target = x[sh[ds:2*ds]]
    yts_target = y1[sh[ds:2*ds]]
    shadow_rep = np.zeros((ns,x.shape[0]-2*ds))
    sh2 = sh[2*ds:]
    
    #Training the target model when size of train & test data = ds
    model = Sequential()
    model.add(Conv2D(nout1, kernel_size, 
                     padding='valid', 
                     input_shape=xtr_target.shape[1:],
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(nout2, kernel_size2, padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(ndense, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    # initiate Adam optimizer
    opt = keras.optimizers.adam(lr=lrate, decay=decay)

    # Let's train the model using Adam
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #print model summary just once
    if i == 0:
        print('Target model summary')
        print(model.summary())
    # Fit the model
    #put verbose = 0 when actually running
    hist_target = model.fit(xtr_target, ytr_target,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(xts_target, yts_target),
                  shuffle=True,verbose=0)
    print('\n\nFor target model with ds = %d'%ds)
    print('Training accuracy = %f'%hist_target.history['acc'][-1])
    print('Validation accuracy = %f'%hist_target.history['val_acc'][-1])
    model_name = 'cifar10_target_'+str(ds)+'.h5'
    model.save(model_name)
    ytemp1 = model.predict(xtr_target)
    ytemp2 = model.predict(xts_target)
    xts_att = np.vstack((ytemp1,ytemp2))
    yts_att = np.zeros(2*ds)
    yts_att[:ds] = 1 
    xts_att_truelabels = np.vstack((ytr_target,yts_target))
    xts_att_dict = {'xts_att':xts_att,'yts_att':yts_att,'xts_att_truelabels':xts_att_truelabels}
    fname = './att_test_data_'+str(ds)
    np.save(fname,xts_att_dict)
    
    xtr_att = np.zeros((2*ds*ns,num_classes))
    ytr_att = np.zeros((2*ds*ns,))
    xtr_att_truelabels = np.zeros((2*ds*ns,num_classes))
    for j in np.arange(ns):
        np.random.shuffle(sh2)
        shadow_rep[j,:] = sh2
        xtr_sh1 = x[sh2[:ds]]
        ytr_sh1 = y1[sh2[:ds]]
        xts_sh1 = x[sh2[ds:2*ds]]
        yts_sh1 = y1[sh2[ds:2*ds]]
        
        model_sh1 = Sequential()
        model_sh1.add(Conv2D(nout1, kernel_size, 
                         padding='valid', 
                         input_shape=xtr_sh1.shape[1:],
                         activation='tanh'))
        model_sh1.add(MaxPooling2D(pool_size=(2, 2)))
        model_sh1.add(Conv2D(nout2, kernel_size2, padding='valid', activation='tanh'))
        model_sh1.add(MaxPooling2D(pool_size=(2, 2)))
        model_sh1.add(Flatten())
        model_sh1.add(Dense(ndense, activation='tanh'))
        model_sh1.add(Dense(num_classes, activation='softmax'))

        # initiate Adam optimizer
        opt_sh1 = keras.optimizers.adam(lr=lrate, decay=decay)

        # Let's train the model using Adam
        model_sh1.compile(loss='categorical_crossentropy',
                          optimizer=opt_sh1,
                          metrics=['accuracy'])
        if j == 0 and i==0:
            print('Shadow model summary:')
            print(model_sh1.summary())
        hist_sh1 = model_sh1.fit(xtr_sh1, ytr_sh1,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(xts_sh1, yts_sh1),
                  shuffle=True,verbose=0)
        model_name = 'cifar10_shadow_'+str(ds)+'_'+str(j)+'.h5'
        model_sh1.save(model_name)
        print('\nFor shadow model %d'%j)
        print('Training accuracy = %f'%hist_sh1.history['acc'][-1])
        print('Validation accuracy = %f'%hist_sh1.history['val_acc'][-1])
        ytemp11 = model_sh1.predict(xtr_sh1)
        ytemp22 = model_sh1.predict(xts_sh1)
        xtr_att[j*2*ds:(j+1)*2*ds] = np.vstack((ytemp11,ytemp22))
        ytr_att[j*2*ds:(2*j+1)*ds] = 1
        xtr_att_truelabels[j*2*ds:(j+1)*2*ds] = np.vstack((ytr_sh1,yts_sh1))
    
    #in outer for loop now
    datafile = './data_cifar10_shadow_'+str(ds)
    np.save(datafile,shadow_rep)
    xtr_att_dict = {'xtr_att':xtr_att,'ytr_att':ytr_att,'xtr_att_truelabels':xtr_att_truelabels}
    fname = './att_train_data_'+str(ds)
    np.save(fname,xtr_att_dict)
#outside both for loops
np.save('./data_cifar10_target',target_rep)


