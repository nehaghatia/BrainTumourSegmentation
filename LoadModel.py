# This file will load the saved model and predict its output
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model,load_model
import tensorflow.keras.callbacks as callbacks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.backend import flatten
import tensorflow as tf
from tensorflow.keras import backend as K

X_train = np.load('./Training Data/X_train.npy')
Y_train = np.load('./Training Data/Y_train.npy')

X_val = np.load('./Validation Data/X_val.npy')
Y_val = np.load('./Validation Data/Y_val.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')

X_test=np.load('./Test Data/X_test.npy')


smooth = 1.
def dice(y_true, y_pred):
    sum_prediction=K.sum(y_pred)
    sum_ground_truth=K.sum(y_true)
    sum_combined=K.sum(y_true * y_pred)
    dice_numerator =2*sum_combined
    dice_denominator =sum_ground_truth+sum_prediction
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score

def dice_coef_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)


model_more_epochs = load_model('Saved Models/modelUnet.h5',custom_objects={'dice_coef_loss': dice_coef_loss, 'dice': dice})

callbacks = [callbacks.EarlyStopping(patience=5, monitor='val_loss')]
history = model_more_epochs.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=5,
                     shuffle=True, callbacks=callbacks)

model_more_epochs.save('Saved Models/newmodelUnet.h5', overwrite=True)
print("ModelSaved Successfully")

# Function to calculate confusion matrix in percentage form
def cal_confusionMatrix(y_valRes, y_preRes):

    print("-------------------------------------------------------------------------")
    print("After converting 1d array reshape y_valRes.shape,y_preRes.shape", y_valRes.shape, y_preRes.shape)
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

    figure2 = sns.heatmap(cf_matirx_classpercent, cmap='Blues', annot=True, fmt='.2%', cbar=False)
    plt.xlabel("Predicted values")
    plt.ylabel("Ground truth values")
    plt.savefig('confusion_matrix')

# Loading of Validation data
X_val = np.load('./Validation Data/X_val.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')

Y_pre_val = np.argmax(model_more_epochs.predict(X_val),axis=-1)
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
print("calculating Whole tumor dice using confusion matrix:",dice_CM_WT)

# Tumor Core
dice_CM_TC = 2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])/(2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])+CM_Y_pre[0,1]+CM_Y_pre[0,3]+CM_Y_pre[2,1]+CM_Y_pre[2,3]+CM_Y_pre[1,0]+CM_Y_pre[1,2:4].sum()+CM_Y_pre[3,0:3].sum())
print("calculating Tumor Core dice using confusion matrix:",dice_CM_TC)

# Enhanced Tumor
dice_CM_ET = 2*(CM_Y_pre[3,3]) / (2*(CM_Y_pre[3,3]) + CM_Y_pre[0:3,3].sum() + CM_Y_pre[3,0:3].sum())
print("calculating Enhanced Tumor dice using confusion matrix:",dice_CM_ET)


for i in range(600, 605):
  print('X_val '+ str(i))
  plt.imshow(X_val[i,:,:,2])
  plt.savefig('X_val ' + str(i))
  plt.show()
  plt.imshow(Y_pre_val[i,:,:,0])
  plt.savefig('Y_pre_val ' + str(i))
  plt.show()
  plt.imshow(Y_val1[i,:,:,0])
  plt.savefig('Y_val ' + str(i))
  plt.show()


