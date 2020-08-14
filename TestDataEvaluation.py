from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model,load_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def dice_coef(y_true, y_pred):
    sum_p=K.sum(y_pred)
    sum_r=K.sum(y_true)
    sum_pr=K.sum(y_true * y_pred)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

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
    plt.savefig('confusion_matrix_percent_testdata_modelOG-HGG-1-2')


# Loading of Test data
X_val = np.load('./Test Data/X_test.npy')
Y_test = np.load('./Test Data/Y_test.npy')

model_more_epochs = load_model('Saved Models/.h5',custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

Y_pre_test = np.argmax(model_more_epochs.predict(X_val),axis=-1)

Y_pre_test=Y_pre_test.reshape(-1,192,192,1)


Y_pre_test = Y_pre_test.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)

y_test_preRes = Y_pre_test.reshape(-1)
Y_testres = Y_test.reshape(-1)

CM_Y_pre = confusion_matrix(Y_testres,y_test_preRes)
print("confusion matrix Test HGG :")
print(CM_Y_pre)

# cal_confusionMatrix(Y_testres,y_test_preRes)

target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(Y_testres, y_test_preRes, target_names=target_names))

# Whole Tumor
Common_FNFP= CM_Y_pre[1:,1:].sum()- (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])
# TP_TPFN = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1:,0:].sum()+Common_FNFP)
TP_TPFN = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1:,0:].sum())
TP_TPFP = (CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3])/(CM_Y_pre[1,1]+CM_Y_pre[2,2]+CM_Y_pre[3,3]+CM_Y_pre[0,1:4].sum())
dice_CM_WT= 2/((1/TP_TPFN)+(1/TP_TPFP))
print("Whole Tumor Dice score on Test data :",dice_CM_WT)

# Tumor Core
dice_CM_TC = 2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])/(2*(CM_Y_pre[1,1]+CM_Y_pre[3,3])+CM_Y_pre[0,1]+CM_Y_pre[0,3]+CM_Y_pre[2,1]+CM_Y_pre[2,3]+CM_Y_pre[1,0]+CM_Y_pre[1,2:4].sum()+CM_Y_pre[3,0:3].sum())
print("Tumor Core Dice score on Test data:",dice_CM_TC)

# Enhanced Tumor
dice_CM_ET = 2*(CM_Y_pre[3,3]) / (2*(CM_Y_pre[3,3]) + CM_Y_pre[0:3,3].sum() + CM_Y_pre[3,0:3].sum())
print("Enhanced Tumor Dice score on Test data:",dice_CM_ET)


for i in range(294, 299):
  print('X_test '+ str(i))
  plt.imshow(X_val[i,:,:,2])
  plt.savefig('X_test ' + str(i))
  plt.show()
  plt.imshow(Y_pre_test[i,:,:,0])
  plt.savefig('Y_pre_test ' + str(i))
  plt.show()
  plt.imshow(Y_test[i,:,:,0])
  plt.savefig('Y_test ' + str(i))
  plt.show()

