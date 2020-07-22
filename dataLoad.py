import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from preprocessing import normalize
import os



path1 = "Data/Brats2018/HGG2/"
path2 = "Data/Brats2018/LGG/"

def load_data(path):
  my_dir = sorted(os.listdir(path))
  # print("length of dir",len(my_dir))  length of dir 75
  # print("my_dir",my_dir) prints the LGG dir files  my_dir ['Brats18_2013_0_1', 'Brats18_2013_15_1', 'Brats18_2013_16_1', 'Brats18_2013_1_1'...]

  data = []
  gt = []

  for p in tqdm(my_dir):
    data_list = sorted(os.listdir(path+p))
    # print("sorted(os.listdir(path+p))",sorted(os.listdir(path+p)))   ['Brats18_2013_0_1_flair.nii.gz', 'Brats18_2013_0_1_seg.nii.gz', 'Brats18_2013_0_1_t1.nii.gz', 'Brats18_2013_0_1_t1ce.nii.gz', 'Brats18_2013_0_1_t2.nii.gz']

    img_itk = sitk.ReadImage(path + p + '/'+ data_list[0])
    # print("image path",path + p + '/'+ data_list[0])  Data/Brats2018/LGG/Brats18_2013_0_1/Brats18_2013_0_1_flair.nii.gz
    flair = sitk.GetArrayFromImage(img_itk)
    # print("flair shape",flair.shape)  # (155, 240, 240)
    # print("flair dtype",flair.dtype)  # int16
    flair = normalize(flair)

    img_itk = sitk.ReadImage(path + p + '/'+ data_list[1])
    seg =  sitk.GetArrayFromImage(img_itk)

    # print("seg shape",seg.shape)  # (155, 240, 240)
    # print("seg dtype",seg.dtype)  # uint8 / int16


    img_itk = sitk.ReadImage(path + p + '/'+ data_list[2])
    t1 =  sitk.GetArrayFromImage(img_itk)
    t1 = normalize(t1)

    img_itk = sitk.ReadImage(path + p + '/'+ data_list[3])
    t1ce =  sitk.GetArrayFromImage(img_itk)
    t1ce = normalize(t1ce)

    img_itk = sitk.ReadImage(path + p + '/'+ data_list[4])
    t2 =  sitk.GetArrayFromImage(img_itk)
    t2 = normalize(t2)

    data.append([flair,t1,t1ce,t2])
    gt.append(seg)

  data = np.asarray(data,dtype=np.float32)
  gt = np.asarray(gt,dtype=np.uint8)
  return data,gt
#
#
# for HGG
data1,gt1 = load_data(path1)  #HGG having 210 patients
# for LGG
# data2,gt2 = load_data(path2)  #LGG having 75 patients

print("data1.shape",data1.shape)
print("gt1.shape",gt1.shape)
print("data1.dtype",data1.dtype)
print("gt1.dtype",gt1.dtype)

#
np.save('HG_data.npy',data1)
np.save('HG_gt.npy',gt1)


