# It contains all the helper function used in the pre-processing of the image before feeding it into the model.
import numpy as np

#  To normalize each brain image
def normalize(img,bottom = 99,top = 1):

    b = np.percentile(img,bottom)
    t = np.percentile(img,top)
    clipImage = np.clip (img,t,b)

    if np.std(clipImage) == 0:
        return clipImage
    else:
        # print("Mean of image",np.mean(clipImage))
        # print("Std of image",np.std(clipImage))
        # print("Min of image", np.min(clipImage))
        # print("Max of image", np.max(clipImage))
        normImage = (clipImage - np.mean(clipImage)) / np.std (clipImage)
        return normImage

# To transpose label of each brain image
def transposeData(data):
    data = np.transpose(data, (0, 2, 3, 4, 1))
    return data

# taking slices 90 slices out of 155 slices
def sliceCrop(data,gt):
    data = data[:,30:120,30:222,30:222,:].reshape([-1,192,192,4])
    gt = gt[:,30:120,30:222,30:222].reshape([-1,192,192,1])
    return data,gt

# converting ground truth value of 4 to 3 to do one hot encoding
def groundTruth4to3(gt):
    gt[np.where(gt==4)]=3
    return gt
