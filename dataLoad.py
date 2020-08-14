import SimpleITK as sitk
from tqdm import tqdm
from preprocessing import *
from sklearn.model_selection import train_test_split
import os
from keras.utils import to_categorical

path1 = "Data/Brats2018/HGG/"
path2 = "Data/Brats2018/LGG/"

# Separating the patients into train, val, test sets:
def load_data(path):
  my_dir = sorted(os.listdir(path))
  print(len(my_dir))

  train, test = train_test_split(my_dir, test_size=0.10, random_state=42)
  train, val = train_test_split(train, test_size=0.25, random_state=42)

  return train, val, test



train, val, test = load_data(path1)
print("Length of train, validate, and test:", len(train), len(val), len(test))


def process_data(path, data):
  data_vector = []
  gt_vector = []

  for p in tqdm(data):
    data_list = sorted(os.listdir(path + p))
    # print("sorted(os.listdir(path+p))",sorted(os.listdir(path+p)))   ['Brats18_2013_0_1_flair.nii.gz',
    # 'Brats18_2013_0_1_seg.nii.gz', 'Brats18_2013_0_1_t1.nii.gz', 'Brats18_2013_0_1_t1ce.nii.gz',
    # 'Brats18_2013_0_1_t2.nii.gz']

    img_itk = sitk.ReadImage(path + p + '/' + data_list[0])
    # print("image path",path + p + '/'+ data_list[0])
    # Data/Brats2018/LGG/Brats18_2013_0_1/Brats18_2013_0_1_flair.nii.gz
    flair = sitk.GetArrayFromImage(img_itk)
    # print("flair shape",flair.shape)  # (155, 240, 240)
    # print("flair dtype",flair.dtype)  # int16
    flair = normalize(flair)

    img_itk = sitk.ReadImage(path + p + '/' + data_list[1])
    seg = sitk.GetArrayFromImage(img_itk)

    # print("seg shape",seg.shape)  # (155, 240, 240)
    # print("seg dtype",seg.dtype)  # uint8 / int16

    img_itk = sitk.ReadImage(path + p + '/' + data_list[2])
    t1 = sitk.GetArrayFromImage(img_itk)
    t1 = normalize(t1)

    img_itk = sitk.ReadImage(path + p + '/' + data_list[3])
    t1ce = sitk.GetArrayFromImage(img_itk)
    t1ce = normalize(t1ce)

    img_itk = sitk.ReadImage(path + p + '/' + data_list[4])
    t2 = sitk.GetArrayFromImage(img_itk)
    t2 = normalize(t2)

    data_vector.append([flair, t1, t1ce, t2])
    gt_vector.append(seg)

  data_vector = np.asarray(data_vector, dtype=np.float32)
  gt_vector = np.asarray(gt_vector, dtype=np.uint8)
  return data_vector, gt_vector


# Training data:
X_train, Y_train = process_data(path1, train)
print("Train data:", len(X_train), len(Y_train))
print("Train shapes:", X_train.shape, Y_train.shape)
print("Train types:", X_train.dtype, Y_train.dtype)
# Validation data:
X_val, Y_val = process_data(path1, val)
print("Val data:", len(X_val), len(Y_val))
print("Val shapes:", X_val.shape, Y_val.shape)
print("Val types:", X_val.dtype, Y_val.dtype)
# Testing data:
X_test, Y_test = process_data(path1, test)
print("Test data:", len(X_test), len(Y_test))
print("Test shapes:", X_test.shape, Y_test.shape)
print("Test types:", X_test.dtype, Y_test.dtype)

X_train = transposeData(X_train)
X_train, Y_train = sliceCrop(X_train, Y_train)

X_val = transposeData(X_val)
X_val, Y_val = sliceCrop(X_val, Y_val)

X_test = transposeData(X_test)
X_test, Y_test = sliceCrop(X_test, Y_test)

Y_train = groundTruth4to3(Y_train)
Y_val = groundTruth4to3(Y_val)
Y_test = groundTruth4to3(Y_test)

# for confusion matrix (without one-hot encoding saving val and train ground truth)
Y_val1=Y_val
Y_train1 = Y_train

print("Before to_categorical")
print("X_train, Y_train shape",X_train.shape, Y_train.shape)
print("X_val, Y_val shape",X_val.shape, Y_val.shape)
print("X_test, Y_test shape",X_test.shape, Y_test.shape)
print("Y_train unique",np.unique(Y_train))

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)


print("Before saving to files data shape")
print("X_train, Y_train shape", X_train.shape, Y_train.shape)
print("X_val, Y_val shape", X_val.shape, Y_val.shape)
print("X_test, Y_test shape", X_test.shape, Y_test.shape)

np.save('./Training Data/X_train.npy',X_train)
np.save('./Training Data/Y_train.npy',Y_train)
np.save('./Validation Data/X_val.npy',X_val)
np.save('./Validation Data/Y_val.npy',Y_val)
np.save('./Test Data/X_test.npy',X_test)
np.save('./Test Data/Y_test.npy',Y_test)

np.save('./Validation Data/Y_val1.npy',Y_val1)
np.save('./Training Data/Y_train1.npy',Y_train1)

print("Data saved successfully")
