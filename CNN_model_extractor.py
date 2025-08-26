import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Flatten, MaxPool3D, Add
from tensorflow.keras.layers import Conv3D, Activation, GlobalAveragePooling3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.layers import Dense, Dropout, Conv3D, Flatten, Activation
from tensorflow.keras.layers import concatenate, BatchNormalization, add, AveragePooling3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model



########Our_simple_model#########  

def CNN_model():
    input_tensor =(64,64,64,1)
    model1 = tf.keras.Sequential([
        layers.Conv3D(32, input_shape = input_tensor, kernel_size = (3,3,3),strides =(1,1,1), padding ='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size = (2,2,2), strides =(2,2,2), padding = 'valid'),
        
        layers.Conv3D(64, kernel_size = (3,3,3), strides =(1,1,1), padding = "same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size =(2,2,2), strides =(2,2,2), padding = 'valid'),
        
        layers.Conv3D(128, kernel_size = (3,3,3), strides =(1,1,1), padding = "same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size =(2,2,2), strides =(2,2,2), padding = 'valid'),
        
        layers.Flatten(),
        layers.Dense(256, activation ='relu'),
        layers.BatchNormalization(),

        # Add a new Dense layer with 32 neurons specifically for feature extraction
        layers.Dense(1024, activation ='relu', name='feature_layer_128'),

        layers.Dense(1, activation = 'sigmoid')
    ])
        
        
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001)
    
    model1.compile(loss= 'binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    
    return model1   

########
def conv_factory(x, nb_filter, kernel=(3, 3, 3), strides=(1, 1, 1),
                 padding='same', dropout_rate=0., weight_decay=0.005):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding=padding,
               strides=strides,
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, growth_rate, internal_layers=4,
               dropout_rate=0., weight_decay=0.005):
    x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    list_feat = []
    list_feat.append(x)
    for i in range(internal_layers - 1):
        x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
    return x

def dense_resnet_3d(weight_decay=0.005, dropout_rate=0.1):

    input_tensor = Input(shape=(64,64,64,1), name='input')

    # stage 1 Initial convolution
    x = Conv3D(64, (3, 3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)


    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(128, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)

    # stage 2
    x = dense_block(x, 32, internal_layers=4,
                             dropout_rate=dropout_rate)
    x = add([x, y])
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)


    # stage 3
    x= dense_block(y, 32, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)


    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(256, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    x1 = conv_factory(x, 128, (1, 1, 2), (2, 2, 2), padding='valid')

    # stage 4
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x2 = conv_factory(y, 128, (1, 1, 1), (1, 1, 1), padding='same')

    # stage 5
    x = dense_block(y, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = concatenate([x, x2, x1], axis=-1)
    x = Conv3D(512, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               strides=(1, 1, 1),
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(1,
              activation='sigmoid',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model1 = Model(inputs=input_tensor, outputs=x)
    model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])

    return model1

########DenSenet
def conv_factory1(x, nb_filter, kernel=(3,3,3), dropout_rate=0., weight_decay=0.005):
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def transition_layer():
    pass


def dense_block(x, growth_rate, internal_layers=4,
               dropout_rate=0., weight_decay=0.005):
    list_feat = []
    list_feat.append(x)
    x = conv_factory1(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    list_feat.append(x)
    x = concatenate(list_feat, axis=-1)
    for i in range(internal_layers - 1):
        x = conv_factory1(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
    return x

def densenet_3d(input_tensor, input_shape,weight_decay=0.005, dropout_rate=0.1):

    input_tensor = Input(shape=(64,64,64,1), name='input')

    # stage 1 Initial convolution
    x = conv_factory1(input_tensor, 64)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    # stage 2
    x = dense_block(x, 32, internal_layers=4,
                             dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory1(x, 128, (1, 1, 1), dropout_rate=dropout_rate)
    # 28x28x4

    # stage 3
    x= dense_block(x, 32, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory1(x, 128, (1, 1, 1), dropout_rate=dropout_rate)

    # 14x14x2

    # stage 4
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory1(x, 256, (1, 1, 1), dropout_rate=dropout_rate)

    # 7x7x1

    # stage 5
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)

    x = conv_factory1(x, 256, (1, 1, 1), dropout_rate=dropout_rate)

    x = GlobalAveragePooling3D()(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model1 = Model(inputs=input_tensor, outputs=x)
       
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
    model1.compile(loss= 'binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    
    return model1 


########Inception
def conv3d_bn(x, nb_filter, kernel=(3, 3, 3), dropout_rate=0., weight_decay=0.005):
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x




def Inception_3d(input_tensor, input_shape, weight_decay=0.005, dropout_rate=0.1):

    input_tensor = Input(shape=(64,64,64,1), name='input')

    x = conv3d_bn(input_tensor, 64, (3, 3, 3))
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    # stage 1
    branch1 = conv3d_bn(x, 32, (1, 1, 1))

    branch2 = conv3d_bn(x, 32, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 32, (5, 5, 3))

    branch3 = conv3d_bn(x, 32, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 32, (3, 3, 3))
    branch3 = conv3d_bn(branch3, 32, (3, 3, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 32, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 28x28x4

    # stage 2
    branch1 = conv3d_bn(x, 32, (1, 1, 1))

    branch2 = conv3d_bn(x, 32, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 32, (5, 5, 3))

    branch3 = conv3d_bn(x, 32, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 32, (3, 3, 3))
    branch3 = conv3d_bn(branch3, 32, (3, 3, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 32, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 14x14x2

    # stage 3
    branch1 = conv3d_bn(x, 64, (1, 1, 1))

    branch2 = conv3d_bn(x, 64, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 64, (5, 5, 3))

    branch3 = conv3d_bn(x, 64, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (7, 1, 3))
    branch3 = conv3d_bn(branch3, 64, (1, 7, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    branch1 = conv3d_bn(x, 64, (1, 1, 1))

    branch2 = conv3d_bn(x, 64, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 64, (5, 5, 3))

    branch3 = conv3d_bn(x, 64, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (7, 1, 3))
    branch3 = conv3d_bn(branch3, 64, (1, 7, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 7x7x1

    # stage 4
    branch1 = conv3d_bn(x, 64, (1, 1, 1))

    branch2 = conv3d_bn(x, 64, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 64, (3, 3, 1))

    branch3 = conv3d_bn(x, 64, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (7, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (1, 7, 1))

    branch4 = AveragePooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    branch1 = conv3d_bn(x, 64, (1, 1, 1))

    branch2 = conv3d_bn(x, 64, (1, 1, 1))
    branch2 = conv3d_bn(branch2, 64, (3, 3, 1))

    branch3 = conv3d_bn(x, 64, (1, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (7, 1, 1))
    branch3 = conv3d_bn(branch3, 64, (1, 7, 1))

    branch4 = AveragePooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv3d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    x = conv3d_bn(x, 256, (1, 1, 1))

    x = GlobalAveragePooling3D()(x)
    x = Dense(1,
              activation='sigmoid',
              kernel_regularizer=l2(0.005),
              bias_regularizer=l2(0.005))(x)
    model = Model(inputs=input_tensor, outputs=x)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
    model.compile(loss= 'binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    return model


#new model
def c3d_model():
    input_shape = (64, 64, 64, 1)
    weight_decay = 0.005
    nb_classes = 1

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPooling3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)
    return model


########
def conv1_layer(x):    
    x = ZeroPadding3D(padding=(3, 3, 3))(x)
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding3D(padding=(1,1,1))(x)
 
    return x   
 
    
def conv2_layer(x):         
    x = MaxPooling3D((3, 3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            shortcut = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv3D(128, (1, 1, 1), strides=(2, 2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv3D(512, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            shortcut = Conv3D(512, (1, 1, 1), strides=(2, 2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv3D(128, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv3D(512, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
 
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv3D(1024, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            shortcut = Conv3D(1024, (1, 1, 1), strides=(2, 2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv3D(1024, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv3D(512, (1, 1, 1), strides=(2, 2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv3D(2048, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            shortcut = Conv3D(2048, (1, 1, 1), strides=(2, 2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv3D(512, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv3D(2048, (1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
#
input_tensor = Input(shape=(64,64,64,1), name='input')
def Resnet_model():
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
 
    x = GlobalAveragePooling3D()(x)   
    output_tensor = Dense(1, activation='sigmoid')(x)
    model1 = Model(input_tensor, output_tensor) 
       
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
    model1.compile(loss= 'binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    
    return model1 


########VGG
def vgg():
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_tensor)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
 
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
     
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
 
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
 
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
 
    x = layers.Flatten()(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    x = layers.Dense(2048, kernel_initializer='he_normal')(x)
    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)
 
    model1 = Model(input_tensor, output_tensor)
    model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    return model1
