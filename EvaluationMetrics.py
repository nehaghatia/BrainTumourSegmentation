from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

 # 2TP/(2TP+FP+FN)
def dice(y_true, y_pred):
    #computes the dice score on two tensors

    sum_p=K.sum(y_pred)
    sum_r=K.sum(y_true)
    # sum_pr --- TP
    sum_pr=K.sum(y_true * y_pred)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score

def dice_coef_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def dice_coef_whole_metric_loss(y_true, y_pred):
    return 1-dice_whole_metric(y_true, y_pred)

def dice_whole_metric(y_true, y_pred):
    #computes the dice for the whole tumor

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))

    # for i in range(6635520):
    # #     if(((y_true_f[i,:]) != np.array([1.,0.,0.,0.])).all()):
    #         print("index, y_true_f values",i,"---------",y_true_f[i,:])

    print(y_true_f.shape, y_pred_f.shape)

    y_whole=y_true_f[:,1:]
    p_whole=y_pred_f[:,1:]

    print(y_whole.shape,p_whole.shape)

    dice_whole=dice(y_whole,p_whole)
    return dice_whole

def dice_en_metric(y_true,y_pred):
    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = y_true.reshape(-1, 4)
    y_pred_f = y_pred.reshape(-1, 4)

    # workaround for tf
    # y_core=K.sum(tf.gather(y_true_f, [1,3]))
    # p_core=K.sum(tf.gather(y_pred_f, [1,3]))
    y_core=y_true_f[:,[1,3]]
    p_core=y_pred_f[:,[1,3]]
    print("shape y core p core", y_core.shape, p_core.shape)
    dice_core = dice(y_core, p_core)
    return dice_core

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
    plt.savefig('confusion_matrix_percent1')