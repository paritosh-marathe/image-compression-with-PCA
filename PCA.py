import math 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# function for image compression using PCA 
file_name='Lenna.png'
def compress(I,numpc=100):
    cov_mat=I-np.mean(I.T,axis=1)
    var=np.cov(I-np.mean(I.T,axis=1))
    eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
    p = np.size(eig_vec, axis =1)
    idx = np.argsort(eig_val)
    idx = idx[::-1]
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]
    if numpc <p and numpc >0:
        eig_vec = eig_vec[:, range(numpc)]
    score = np.dot(eig_vec.T, cov_mat)
    return eig_val,eig_vec,score


img=cv2.imread(file_name)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
numpc =50 # number of principle components

eig_val,eig_vec,score=compress(img ,numpc)
recon = np.dot(eig_vec, score) + np.mean(img.T, axis = 1) 
recon_img_mat = np.uint8(np.absolute(recon)) # reconstructed image from eigens

cv2.imwrite("recon_image.jpg",recon_img_mat)
mse=np.sum((img.astype('float')-recon_img_mat.astype('float'))**2)/float(255*255)#/float(I.shape[0]*recon_img_mat.shape[1])
r=65025/mse
PSNR=10* math.log(r,10)
print(' The MSE is: ',mse)
#print('The PSNR is {} dB'.format(r))