#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Affine Translation
##we use cv2.warpaffine to implement these transformation

##Rotaions
##cv2.getRotationMatrix2D(rotation_center_x,rotation_center_y,angle_of_rotation,scale)


# In[2]:


import os
os.getcwd()


# In[3]:


os.chdir('D:\Learnbay\Computer Vision\images')


# In[4]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('input.jpg')

# Store height and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width =height/4, width/4

# Translation matrix(T) = |1 0 Tx|
#                         |0 1 TY|

T = np.float32([[1,0, quarter_width],[0,1,quarter_height]])

# we use warpAffine to transform the image using the matrix, T

img_translation = cv2.warpAffine(image, T, (width, height))

cv2.imshow('Translation Image', img_translation)
cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.destroyAllWindows()

#https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html


# In[5]:


image.shape


# In[6]:


image.shape[:2]


# In[7]:


print(T)


# In[ ]:





# ## Rotation

# In[8]:


# cv2.getRotationMatrix2D()

image =cv2.imread('input.jpg')

height, width = image.shape[:2]

#Divide by two to rotate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2),90,0.5)
rotated_image = cv2.warpAffine(image,rotation_matrix,(width, height))
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Translation Image', img_translation)
cv2.imshow('Original Image',image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[9]:


rotated_image.shape


# In[10]:


image = cv2.imread('input.jpg')
rotated_image =cv2.transpose(image)
cv2.imshow('Rotated Image-method 2', rotated_image)
cv2.imshow('Translation Image', img_translation)
cv2.imshow('Original Image',image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[11]:


flipped =cv2.flip(image, 1)
cv2.imshow('Rotated Image-method 2', rotated_image)
cv2.imshow('Horizontal Flip',flipped)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# # Re-sizing, Scaling and Interpolation

# In[12]:


image = cv2.imread('input.jpg')

cv2.imshow('Original Image',image)
cv2.waitKey()

#Let's make the above image as 3/4 of it's original size

image_scaled =cv2.resize(image, None, fx = 0.75, fy =0.75)
cv2.imshow('Scaling-Linear Interpolation', image_scaled)
cv2.waitKey()

image_scaled1 = cv2.resize(image, None, fx =2, fy=2, interpolation =cv2.INTER_CUBIC)
cv2.imshow('scaling -Cubic Interpolation',image_scaled1)
cv2.waitKey()

cv2.destroyAllWindows()


# In[13]:


def show_pic(img):
    fig = plt.figure(figsize = (20,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap ='gray')


# In[14]:


img =cv2.imread('crossword.PNG',0)  # 0 = grayscale image
plt.imshow(img,cmap ='gray')


# In[15]:


show_pic(img)


# In[16]:


ret, img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show_pic(img1)
# here we have more clarity, but the picture breaking


# In[17]:


image = cv2.imread('input.jpg')

cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.destroyAllWindows()


#  # Pyramids

# In[19]:


image = cv2.imread('input.jpg')

smaller = cv2.pyrDown(image) # reducing the size to half

larger = cv2.pyrUp(image) # increasing the size to double

cv2.imshow('Original Image', image)
cv2.imshow('Smaller', smaller)
cv2.imshow('Larger', larger)

cv2.waitKey()
cv2.destroyAllWindows()


# # Cropping

# In[23]:


image = cv2.imread('input.jpg')
height, width =image.shape[:2]

start_row,start_col = int(height * 0.25),int(width * 0.25)
end_row,end_col = int(height *0.75), int(width * 0.75)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("Original Image", image)
cv2.waitKey()

cv2.imshow("Cropped Image", cropped)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




