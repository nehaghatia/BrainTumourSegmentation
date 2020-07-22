import numpy as np
from tensorflow.keras.utils import to_categorical
from EvaluationMetrics import confusion_matrix

def validation(model):
    X_val = np.load('./Validation Data/X_val4.npy')
    Y_val = np.load('./Validation Data/Y_val4.npy')
    Y_val1 = np.load('./Validation Data/Y_val1.npy')

    Y_pre_val = np.argmax(model.predict(X_val),axis=-1)
    np.save('./Prediction Data/Y_pre_val_with_normalise.npy',Y_pre_val)

    # Confusion matrix for Validation cases
    X_test = np.load('./Validation Data/X_val4.npy')
    Y_test = np.load('./Validation Data/Y_val1.npy')
    Y_pre = np.load('./Prediction Data/Y_pre_val_with_normalise.npy')


    print("X_val.shape", "Y_val1.shape", "Y_pre_val.shape", X_val.shape, Y_val1.shape, Y_pre_val.shape)
    confusion_matrix(Y_pre,Y_val1)
