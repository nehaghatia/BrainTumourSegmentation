# Code to develop ResUnet architecture
import numpy as np
from keras.backend import flatten
from keras.layers import Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,add,concatenate,Input,Dropout,Maximum,Activation,Dense,Flatten,UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

X_train = np.load('./Training Data/X_train.npy')
Y_train = np.load('./Training Data/Y_train.npy')

X_val = np.load('./Validation Data/X_val.npy')
Y_val = np.load('./Validation Data/Y_val.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')

# Model code is taken from
# https://github.com/nikhilroxtomar/Deep-Residual-Unet
def batchnorm_activation(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x


def convolution_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = batchnorm_activation(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def initialBlock(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = convolution_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = batchnorm_activation(shortcut, act=False)

    output = Add()([conv, shortcut])
    return output


def resblock(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = convolution_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = convolution_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = batchnorm_activation(shortcut, act=False)

    output =Add()([shortcut, res])
    return output


def upsampleConBlock(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet():
    f = [ 32,64, 128,256,512]
    inputs = Input((192, 192, 4))

    e0 = inputs
    e1 = initialBlock(e0, f[0])
    e2 = resblock(e1, f[1], strides=2)
    e3 = resblock(e2, f[2], strides=2)
    e4 = resblock(e3, f[3], strides=2)


    b0 = resblock(e4, f[4], strides=2)

    u1 = upsampleConBlock(b0, e4)
    d1 = resblock(u1, f[4])

    u2 = upsampleConBlock(d1, e3)
    d2 = resblock(u2, f[3])

    u3 = upsampleConBlock(d2, e2)
    d3 = resblock(u3, f[2])

    u4 = upsampleConBlock(d3, e1)
    d4 = resblock(u4, f[1])

    outputs =Conv2D(4, (1, 1), padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs)
    return model



smooth = 1.

def dice_coef(y_true, y_pred):
    # based on confusion matrix and targeting whole tumour. This way rejecting background from dice coef calculation and hence loss is specific to tumours

    # print(y_true.shape,y_pred.shape)
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    # loss1= tf.subtract(y_true,y_pred,name=None)
    return 1- dice_coef(y_true,y_pred)

model = ResUNet()

# model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice])
model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice_coef])
# model.load_weights('./Model Checkpoints/weights.hdf5')
# checkpointer = callbacks.ModelCheckpoint(filepath = './Model Checkpoints/weights.hdf5',save_best_only=True)
# training_log = callbacks.TensorBoard(log_dir='./Model_logs')


early_stopping = [callbacks.EarlyStopping(patience=5, monitor='val_loss')]

# history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=32,epochs=16,callbacks=[checkpointer],shuffle=True)
history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=16,epochs=5,shuffle=True,callbacks=[early_stopping])

model.save('Saved Models/DeepResidualUnet-HGG_5epochs.h5',overwrite=True)
# print(model.summary())
# model_more_epochs.save('Saved_models/modelOG-HGG-1-3.h5', overwrite=True)
# print("ModelSaved Successfully")


def cal_confusionMatrix(y_valRes, y_preRes):

    print("-------------------------------------------------------------------------")
    print("After converting 1d array reshape y_valRes.shape,y_preRes.shape", y_valRes.shape, y_preRes.shape)
    # After converting 1d array reshape y_testRes.shape,y_preRes.shape (26542080,) (26542080,)
    CM_Y_pre = confusion_matrix(y_valRes, y_preRes)
    print("confusion matrix Validation :")
    print(CM_Y_pre)


    cf_matrix = confusion_matrix(y_valRes, y_preRes)
    print("cf_matrix shape", cf_matrix.shape)

    cf_matirx_classpercent = np.zeros(shape=(4, 4))
    for j in range(4):
        for i in range(4):
            cf_matirx_classpercent[j][i] = cf_matrix[j][i] / (
                        cf_matrix[j][0] + cf_matrix[j][1] + cf_matrix[j][2] + cf_matrix[j][3])
            print(cf_matirx_classpercent[j][i])

    print("cf_matirx_classpercent.shape", np.shape(cf_matirx_classpercent))
    figure2 = sns.heatmap(cf_matirx_classpercent, cmap='Blues', annot=True, fmt='.2%', cbar=False)
    plt.xlabel("Predicted values")
    plt.ylabel("Ground truth values")
    plt.savefig('DeepResidualUnet-HGG_5epochs')

# Loading of Validation data
X_val = np.load('./Validation Data/X_val.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')

Y_pre_val = np.argmax(model.predict(X_val),axis=-1)
np.save('./Prediction Data/Y_pre_val_HGG.npy',Y_pre_val)

Y_pre_val=Y_pre_val.reshape(-1,192,192,1)

Y_pre_val = Y_pre_val.astype(np.uint8)
Y_val1 = Y_val1.astype(np.uint8)

y_val_preRes = Y_pre_val.reshape(-1)
y_valRes = Y_val1.reshape(-1)

CM_Y_pre = confusion_matrix(y_valRes,y_val_preRes)
print("confusion matrix Validate :")
print(CM_Y_pre)
cal_confusionMatrix(y_valRes,y_val_preRes)

target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(y_valRes, y_val_preRes, target_names=target_names))

#calculating dice using confusion matrix
# TP_TPFN is recall and TP_TPFP is precission and DICE is HM of these two.
# This equation is also = 2TP/(2TP+FP+FN) after simplification
# Whole Tumor
Common_FNFP= CM_Y_pre[1:,1:].sum()- (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])
# TP_TPFN = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1:,0:].sum()+Common_FNFP)
TP_TPFN = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1:,0:].sum())
TP_TPFP = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3]+CM_Y_pre[0,1:4].sum())
dice_CM_WT= 2/((1/TP_TPFN)+(1/TP_TPFP))
print("Whole tumor dice :",dice_CM_WT)

# Tumor Core
dice_CM_TC = 2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])/(2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])+CM_Y_pre[0,1]+CM_Y_pre[0,3]+CM_Y_pre[2,1]+CM_Y_pre[2,3]+CM_Y_pre[1,0]+CM_Y_pre[1,2:4].sum()+CM_Y_pre[3,0:3].sum())
print("Tumor Core dice :",dice_CM_TC)

# Enhanced Tumor
dice_CM_ET = 2*(CM_Y_pre[3,3]) / (2*(CM_Y_pre[3,3]) + CM_Y_pre[0:3,3].sum() + CM_Y_pre[3,0:3].sum())
print("Enhanced Tumor dice :",dice_CM_ET)

# For displaying predicted images
# for i in range(600, 605):
#   print('X_val '+ str(i))
#   plt.imshow(X_val[i,:,:,2])
#   plt.savefig('X_val ' + str(i))
#   plt.show()
#   plt.imshow(Y_pre_val[i,:,:,0])
#   plt.savefig('Y_pre_val ' + str(i))
#   plt.show()
#   plt.imshow(Y_val1[i,:,:,0])
#   plt.savefig('Y_val ' + str(i))
#   plt.show()








