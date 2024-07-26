#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[16]:


def conv_block(input, num_filters):
  x = tkl.Conv2D(num_filters, 3, padding='same')(input)
  x = tkl.Activation('relu')(x)
    
  x = tkl.Conv2D(num_filters, 3, padding='same')(input)
  x = tkl.Activation('relu')(x)
  return x


# In[17]:


def encoder_block(input, num_filters):
    
  x = conv_block(input, num_filters)
  p = tkl.MaxPool2D((2, 2))(x)
    
  return x, p


# In[18]:


def decoder_block(input, skip_features, num_filters):
  x = tkl.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
  x = tkl.Concatenate()([x, skip_features])
  x = conv_block(x, num_filters)
  return x


# In[24]:
def gatting(input,num_filters):
    
  x = tkl.Conv2D(num_filters, (1,1) ,strides = 2 , padding='same')(input)

  x = tkl.Activation('relu')(x) 
  return x  


# # In[21]:


def attention_layer(input,input_to_gat):
    
    num_filters = input.shape[-1]#16x16x512
    
    input_size = input.shape[:-1]
    
    gated_sig = gatting(input_to_gat,num_filters)
    
    x = tkl.Multiply()([gated_sig, input])
    
    x = tkl.Activation('relu')(x) 

    x = tkl.Conv2D(1,(1,1),strides = 1,padding ='same')(x)
    
    x = tkl.Activation('sigmoid')(x)
    
    x = tkl.UpSampling2D()(x)
    
    x = tkl.Multiply()([input_to_gat, x])
    return x

def build_attention_unet(input_shape,num_classes):
    
    
     inputs = tkl.Input(input_shape)
        
     s1, p1 = encoder_block(inputs, 64)
    
     s2, p2 = encoder_block(p1, 128)
        
     s3, p3 = encoder_block(p2, 256)
    
     s4, p4 = encoder_block(p3, 512)

     b1 =  conv_block(p4, 1024)#base layer output
    
     d1 =  attention_layer(b1,s4)
        
     d1 =  decoder_block(b1, s4, 512)
    
     d2 =  attention_layer(d1,s3)
     d2 =  decoder_block(d1, s3, 256)
    
     d3 =  attention_layer(d2,s2)
     d3 =  decoder_block(d2, s2, 128)
    
     d4 =  attention_layer(d3,s1)
     d4 =  decoder_block(d3, s1, 64)

     outputs = tkl.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)  #soft_max for multi-class
     model = tf.keras.Model(inputs, outputs, name='UNET')

     return model


# In[25]:


model = build_attention_unet((256,256,3),1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[22]:



# In[ ]:




