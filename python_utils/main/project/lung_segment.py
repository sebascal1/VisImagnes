from skimage import filters, morphology, exposure
from skimage.morphology import binary_closing
from skimage.morphology import disk
from skimage.morphology import binary_opening
from scipy import stats
from skimage import measure
import cv2
from tqdm.auto import tqdm
import nibabel as nib 
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

def interThresh(img, minLevel, maxLevel, dtype = "float32"):
    dim = img.shape[0]
    imgOut = img
    for i in range(dim):
        for j in range(dim):
            if img[i, j] <= minLevel:
                imgOut[i,j] = minLevel
            elif img[i,j] >= maxLevel:
                imgOut[i,j] = maxLevel
    return imgOut.astype(dtype)

def blackBackground(img):
  dim = img.shape[0]
  imgOut = img
  ##Upper part
  for i in range(round((100 / 512) * dim)):
    for j in range(dim):
      imgOut[i,j] = 0   

  ##middle (lung part)
  for i in range(round((100 / 512) * dim), round((420 / 512) * dim)):
    for j in range(dim):
      if img[i,j] == 1.0:
        imgOut[i, j] = 0
      elif img[i,j] == 0:
        break

  for i in range(round((100 / 512) *dim), round((420 / 512) * dim)):
    for j in reversed(range(dim)):
      if img[i,j] == 1.0:
        imgOut[i, j] = 0
      elif img[i,j] == 0:
        break

  ##lower part
  for i in range(round((440 / 512) * dim), dim):
    for j in range(dim):
      imgOut[i,j] = 0   
  return imgOut

def findLabelsAlt(img):
    all_labels = pd.Series(img.ravel())
    labelCounts = all_labels.value_counts().index.tolist()
    label1 = labelCounts[1]

    if (len(labelCounts) < 3):
        print("labelcounts smaller than 3")
        label2 = label1
    else:
        label2 = labelCounts[2]
    return label1, label2
 
def getLungs(img, col1, col2):
  dim = img.shape[0]
  imgOut = np.empty(shape = (dim,dim))
  for i in range(dim):
    for j in range(dim):
      if (img[i, j] == col1) or (img[i, j] == col2):
        imgOut[i, j] = 1
      else:
        imgOut[i, j] = 0
  return imgOut
  
def boolToNum(img):
  dim = img.shape[0]
  imgOut = np.empty(shape = (dim,dim))
  for i in range(dim):
    for j in range(dim):
      if (img[i,j]):
        imgOut[i,j] = 1
      else:
         imgOut[i,j] = 0
  return imgOut.astype(int)
  
def centerFill(img):
  dim = img.shape[0]
  for i in range(dim):
    for j in range(dim):
      if ((j >= round(150 * dim/512)) and (j <= round(350 * dim/512))) and ((i >= round(180 * dim/512)) and (i <= round(380 * dim/512))):
        if img[i, j] != 1:
          img[i, j] = 1
  return img
  
def fillLungs(img):
  dim = img.shape[0]
  #Preliminary fill of region between lungs
  for i in range(dim-1):
    colChange = 0
    for j in range(dim-1):
      #Check whether the next pixel is the same color or not
      #if not, write down instance
      if img[i, j+1] != img[i, j]:
        colChange = colChange + 1
        ## if equal to 2 changes, save location of edge
        if colChange == 2:
          edge1 = j
        ##if it is equal to 3, means extra edge has been found
        #fill the inside
        elif colChange == 3:
          for z in range(j - edge1 +1):
            img[i, z + edge1] = 1
          j = dim-2

  #After preliminary fill, morphological closing to close spaces between lines
  img = binary_closing(img, morphology.disk(6))
  img = binary_closing(img, morphology.disk(6))
  return img
  
def lungSegmentPipeline(img, annot, thresh, i):

  #Image binarization
  #imgBinary = img < thresh
  imgBinaryHigh = img >= thresh
  imgBinaryLow = img <= thresh
  imgBinaryFill = ndimage.binary_fill_holes(imgBinaryHigh).astype(int)
  imgBinary = imgBinaryFill * imgBinaryLow

  #set background to black for easier image extraction
  #imgBinary = blackBackground(imgBinary)
  #remove noise using closing and opening binary operations
  imgBinary = binary_opening(imgBinary)
  imgBinary = binary_closing(imgBinary)
  #get connected components
  labels = measure.label(imgBinary, background=0)
  #extract labels relating to lungs
  lab1, lab2 = findLabelsAlt(labels)
  #extract lungs from image and perform morphological closing
  lungs = getLungs(labels, lab1, lab2)
  lungs = binary_closing(lungs, morphology.disk(6))
  #turn output boolean image to true/false
  lungs = boolToNum(lungs)
  #bitwise closing between lung area
  lungs = fillLungs(lungs)
  lungs = fillLungs(lungs)
  lungs = boolToNum(lungs)
  lungs = ndimage.binary_fill_holes(lungs).astype(int)
  #apply binary mask to get segmented lung image
  lungs = lungs * img
  lungs = lungs.astype("float32")
  #lungs = lungs.astype("uint8")
  #print lung image
  #labeled_image = get_labeled_image(lungs, annot, 3)
  #plt.figure(figsize=(6, 6))
  #imshow(labeled_image, title=i)
  return lungs
