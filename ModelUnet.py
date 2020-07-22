import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,concatenate,Input,Dropout,Maximum,Activation,Dense,Flatten,UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks
# import keras.initializers as initializers
# from keras.callbacks import Callback
# from keras import regularizers
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from EvaluationMetrics import *

X_train = np.load('./Training Data/X_train4.npy')
Y_train = np.load('./Training Data/Y_train4.npy')

X_val = np.load('./Validation Data/X_val4.npy')
Y_val = np.load('./Validation Data/Y_val4.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')


input_ = Input(shape=(192,192,4),name='input')

block1_conv1 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv1')(input_)
block1_conv2 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv2')(block1_conv1)
block1_norm = BatchNormalization(name='block1_batch_norm')(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

encoder_dropout_1 = Dropout(0.2,name='encoder_dropout_1')(block2_pool)

block3_conv1 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv1')(encoder_dropout_1)
block3_conv2 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)
################### Encoder end ######################

block5_conv1 = Conv2D(1024,3,padding='same',activation='relu',name='block5_conv1')(block4_pool)
# encoder_dropout_2 = Dropout(0.2,name='encoder_dropout_2')(block5_conv1)

########### Decoder ################

up_pool1 = Conv2DTranspose(1024,3,strides = (2, 2),padding='same',activation='relu',name='up_pool1')(block5_conv1)
merged_block1 = concatenate([block4_norm,up_pool1],name='merged_block1')
decod_block1_conv1 = Conv2D(512,3, padding = 'same', activation='relu',name='decod_block1_conv1')(merged_block1)

up_pool2 = Conv2DTranspose(512,3,strides = (2, 2),padding='same',activation='relu',name='up_pool2')(decod_block1_conv1)
merged_block2 = concatenate([block3_norm,up_pool2],name='merged_block2')
decod_block2_conv1 = Conv2D(256,3,padding = 'same',activation='relu',name='decod_block2_conv1')(merged_block2)

decoder_dropout_1 = Dropout(0.2,name='decoder_dropout_1')(decod_block2_conv1)

up_pool3 = Conv2DTranspose(256,3,strides = (2, 2),padding='same',activation='relu',name='up_pool3')(decoder_dropout_1)
merged_block3 = concatenate([block2_norm,up_pool3],name='merged_block3')
decod_block3_conv1 = Conv2D(128,3,padding = 'same',activation='relu',name='decod_block3_conv1')(merged_block3)

up_pool4 = Conv2DTranspose(128,3,strides = (2, 2),padding='same',activation='relu',name='up_pool4')(decod_block3_conv1)
merged_block4 = concatenate([block1_norm,up_pool4],name='merged_block4')
decod_block4_conv1 = Conv2D(64,3,padding = 'same',activation='relu',name='decod_block4_conv1')(merged_block4)
############ Decoder End ######################################

# decoder_dropout_2 = Dropout(0.2,name='decoder_dropout_2')(decod_block4_conv1)

pre_output = Conv2D(64,1,padding = 'same',activation='relu',name='pre_output')(decod_block4_conv1)

output = Conv2D(4,1,padding='same',activation='softmax',name='output')(pre_output)

model = Model(inputs = input_, outputs = output)
print(model.summary())



model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice])
# model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_whole_metric_loss,metrics=[dice_whole_metric])
# model.load_weights('./Model Checkpoints/weights.hdf5')
# checkpointer = callbacks.ModelCheckpoint(filepath = './Model Checkpoints/weights.hdf5',save_best_only=True)
# training_log = callbacks.TensorBoard(log_dir='./Model_logs')

# history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=32,epochs=16,callbacks=[checkpointer],shuffle=True)
history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=16,epochs=16,shuffle=True)


# Prediction
# Y_pre = np.argmax(model.predict(X_test),axis=-1)
# np.save('./Prediction Data/Y_pre_with_normalise.npy',Y_pre)

Y_pre_val = np.argmax(model.predict(X_val),axis=-1)
np.save('./Prediction Data/Y_pre_val_with_normalise.npy',Y_pre_val)

# Y_pre_train = np.argmax(model.predict(X_train),axis=-1)
# np.save('./Prediction Data/Y_pre_train_with_normalise.npy',Y_pre_train)

Y_pre_val = Y_pre_val.astype(np.uint8)
Y_val1 = Y_val1.astype(np.uint8)

y_val_preRes = Y_pre_val.reshape(-1)
y_valRes = Y_val1.reshape(-1)

cal_confusionMatrix(y_valRes,y_val_preRes)

# Encoding one hot
Y_pre_val = to_categorical(Y_pre_val)
Y_val1 = to_categorical(Y_val1)
print("While calculating Whole Tumour X_val, Y_val1.shape, Y_pre.shape",X_val.shape,Y_val1.shape,Y_pre_val.shape)
Dice_Whole = dice_whole_metric(Y_val1,Y_pre_val)
Dice_Core = dice_core_metric(Y_val1,Y_pre_val)
Dice_Enhanced = dice_en_metric(Y_val1,Y_pre_val)
print("Dice_Whole value for Validate",Dice_Whole)
print("Dice_Core value for Validate",Dice_Core)
print("Dice_Enhanced value for Validate",Dice_Enhanced)



# Try checking for Whole Tumour

