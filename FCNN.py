import keras
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D,Input,Multiply,RepeatVector,TimeDistributed,Add,Concatenate,Lambda,Reshape,Average
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.metrics import categorical_accuracy
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras import backend as K
from keras.losses import CategoricalCrossentropy
from tensorflow import split,divide



l2_regularization=0.5
regularization = l2(l2_regularization)

#************************************************
#confidence stuff

def confidence_loss(y_true, y_pred):
    l=-K.log(y_pred)
    #K.zeros_like(l)
    return l

def accuracy_without_conf(yTrue,yPred):
    yT = yTrue[:,0:7]
    yP = yPred[:,0:7]
    return categorical_accuracy(yT,yP)

lmd=K.variable(0.2,name='example_var')

def loss_with_conf(conf):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        new_pred=conf*y_pred+(1-conf)*y_true
        l=K.categorical_crossentropy(new_pred,y_true)
        return l
    # Return a function
    return loss

def slice_layer(x,i):
    x=x[:,:,:,i]
    x=K.expand_dims(x,axis=-1)
    return x

def get_half(x,i):
    halves=split(value=x,num_or_size_splits=2,axis=2)
    return halves[i]


def get_FCNN_attention_new(num_channels=1,filt_size=3,att_filt_size=16):
    img_input = Input((None,None,num_channels))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=att_filt_size, kernel_size=(filt_size,filt_size),strides=(1,1))(x)
    x=Activation('relu')(x)
    print(x.shape)
    print('before')
    
    
    mag=Lambda(get_half,arguments={'i':0})(x)
    mag_att=Conv2D(filters=1, kernel_size=(filt_size,filt_size),strides=(1,1),padding='same',kernel_regularizer=regularization)(mag)
    mag_att=Activation('sigmoid')(mag_att)
    mag_att=Concatenate()([mag_att]*att_filt_size)
    mag_mul=Multiply(name='mag_attention')([mag,mag_att])
    mag_mul=Activation('relu')(mag_mul)
    
    mag=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(mag_mul)
    mag=GlobalAveragePooling2D()(mag)

    
    mgd=Lambda(get_half,arguments={'i':1})(x)
    mgd_att=Conv2D(filters=1, kernel_size=(filt_size,filt_size),strides=(1,1),padding='same',kernel_regularizer=regularization)(mgd)
    mgd_att=Activation('sigmoid')(mgd_att)
    mgd_att=Concatenate()([mgd_att]*att_filt_size)
    mgd_mul=Multiply(name='mgd_attention')([mgd,mgd_att])
    mgd_mul=Activation('relu')(mgd_mul)
    
    mgd=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(mgd_mul)
    mgd=GlobalAveragePooling2D()(mgd)
    
    x=Average()([mag,mgd])    
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def get_FCNN_RAVDESS(num_classes):
    img_input = Input((None,None,1))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=16, kernel_size=(6,6),strides=(1,1))(x)
    x=Activation('relu')(x) 
    
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=num_classes, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd=optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def get_FCNN_bn_1():
    img_input = Input((None,None,1))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=16, kernel_size=(6,6),strides=(1,1))(x)
    x=Activation('relu')(x) 
    
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd=optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
def get_FCNN_bn_2():
    img_input = Input((None,None,1))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=16, kernel_size=(6,6),strides=(1,1))(x)
    x=Activation('relu')(x) 
    
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd=optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
def get_FCNN_bn_3():
    img_input = Input((None,None,1))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=16, kernel_size=(6,6),strides=(1,1))(x)
    x = BatchNormalization()(x)
    x=Activation('relu')(x) 
    
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd=optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def get_FCNN_attention(num_filt1=32,num_filt2=64,num_channels=1,filt_size=6,att_filt_size=16):
    img_input = Input((None,None,num_channels))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    #x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=att_filt_size, kernel_size=(filt_size,filt_size),strides=(1,1))(x)
    x=Activation('relu')(x)
    #attention module
    att=Conv2D(filters=1, kernel_size=(filt_size,filt_size),strides=(1,1),padding='same',kernel_regularizer=regularization)(x)
    att = Activation('sigmoid')(att)
    att=Concatenate()([att]*att_filt_size)
    mul=Multiply(name='attention')([x,att])
    x=Activation('relu')(mul)
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
    

def get_confidence2_FCNN():    
    img_input = Input((None,None,1))   
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    

    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x) 

    num_filters=18
    x=Conv2D(filters=num_filters, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    
    #********************
    #attention mechanism
    l=[]
    y_list=[]
    for i in range(0,num_filters):
        y=Lambda(slice_layer,arguments={'i':i})(x)
        y=Conv2D(filters=1,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularization)(y)
        y=Activation('sigmoid')(y)
        y_list.append(y)
        l.append(y)
    
    conc=l[0]
    for i in range(1,num_filters):
        conc=Concatenate()([conc,l[i]]) 
        
    x=Multiply(name='attention')([x,conc])
    x=Activation('relu')(x)
    print(x.shape)    
    #************************************

    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='prediction')(x)


    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(optimizer=adam, loss=[loss_with_conf(conf),confidence_loss],loss_weights=[1,lmd],metrics=[accuracy_without_conf])
    model.compile(optimizer=adam, loss=['categorical_crossentropy'],metrics=['accuracy'])
    return model

def get_confidence_FCNN():
    lmd=K.variable(0.2,name='example_var')
    
    img_input = Input((None,None,1))   
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    

    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  

    x=Conv2D(filters=16, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    
    #attention
    att=Conv2D(filters=1, kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularization)(x)
    att=Activation('sigmoid')(att)
    conc=Concatenate()([att]*16)
    x=Multiply(name='attention')([x,conc])
    
    conf=GlobalAveragePooling2D()(att)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,[output,conf])
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss=[custom_loss(conf),confidence_loss],loss_weights=[1,0],metrics=[accuracy_without_conf])
    return model,lmd


def get_FCNN_attention_exp(num_classes=7,num_channels=1,filt_size=6,att_filt_num=16):
    img_input = Input((None,None,num_channels))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=att_filt_num, kernel_size=(filt_size,filt_size),strides=(1,1))(x)
    x=Activation('relu')(x)
    #attention module
    att=Conv2D(filters=att_filt_num, kernel_size=(filt_size,filt_size),strides=(1,1),padding='same',kernel_regularizer=regularization)(x)
    att = Activation('sigmoid')(att)
    mul=Multiply(name='attention')([x,att])
    x=Activation('relu')(mul)
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=num_classes, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model



def get_FCNN_attention_classifier(num_channels):
    img_input = Input((None,None,num_channels))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=2, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy']) 
    return model


#with no attention layer
def get_FCNN(num_channels=1,filt_size=3,filt_num=16):
    img_input = Input((None,None,num_channels))
    
    x=Conv2D(filters=32, kernel_size=(3,3),strides=(1,1))(img_input)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)    
    
    x=Conv2D(filters=64, kernel_size=(3,3),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)  
    
    x=Conv2D(filters=filt_num, kernel_size=(filt_size,filt_size),strides=(1,1))(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Conv2D(filters=7, kernel_size=(3,3),strides=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input,output)
    adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model


