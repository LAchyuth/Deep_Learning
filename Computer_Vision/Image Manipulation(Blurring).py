#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import numpy as np
import pandas as pd
import cv2


# In[16]:


os.getcwd()


# In[17]:


os.chdir('D:\Learnbay\Computer Vision\images')


# In[18]:


image =cv2.imread('elephant.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[19]:


# Here we are blurring the image

image =cv2.imread('elephant.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey()

#creating our 3*3 kernel (convolution)
kernel_3x3 = np.ones((3,3),np.float32)/9

# we use the cv2.filtered2D to convolve the kernel with original image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3*3 kernel blurring', blurred)
cv2.waitKey()

cv2.destroyAllWindows()


# In[20]:


# Increase the blurness

image =cv2.imread('elephant.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey()

#creating our 3*3 kernel (convolution)
kernel_8x8 = np.ones((8,8),np.float32)/64

# we use the cv2.filtered2D to convolve the kernel with original image
blurred = cv2.filter2D(image, -1, kernel_8x8)
cv2.imshow('8*8 kernel blurring', blurred)
cv2.waitKey()

cv2.destroyAllWindows()


# In[21]:


blur = cv2.blur(image, (8,8))
cv2.imshow('Avg Blur',blur)
cv2.waitKey()
cv2.destroyAllWindows()


# In[22]:


# Instead of using other blur, we prefer to use Gaussian Kernel

gaussian =cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blur', gaussian)
cv2.waitKey()

cv2.imshow('7*7 kernel blurring', blurred)
cv2.waitKey()
cv2.destroyAllWindows()

# in this main image will be better than background compared to other types


# In[23]:


median = cv2.medianBlur(image,5)
cv2.imshow('median blurring',median)
cv2.waitKey()
cv2.destroyAllWindows()


# In[24]:


bilateral = cv2.bilateralFilter(image,25,100,100)
cv2.imshow('bilateral blurring',bilateral)
cv2.waitKey()
cv2.destroyAllWindows()


# In[25]:


dst = cv2.fastNlMeansDenoisingColored(image, None,6,6,7,21)
cv2.imshow('fastNlMeansDenoisingColored',dst)
cv2.waitKey()
cv2.destroyAllWindows()


# # Sharpening(More Clarity)

# In[26]:


image =cv2.imread('mask.jpg')
cv2.imshow('Original Image', image)

kernel_sharpening = np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
cv2.imshow("Sharpen Image", sharpened)
cv2.waitKey()
cv2.destroyAllWindows()


# In[27]:


# Increasing the quality of the image

image1 =cv2.imread('mask.jpg')
cv2.imshow('Original Image', image1)

kernel_sharpening = np.array([[-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1],
                             [-1,-1,25,-1,-1],
                             [-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1]])
sharpened = cv2.filter2D(image1, -1, kernel_sharpening)
cv2.imshow("Sharpen Image", sharpened)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# # Threshold

# In[28]:


image =cv2.imread('gradient.jpg')
cv2.imshow('Original Image', image)

# Values below 127 goes to 0 and above 127 goes to 255
ret, thresh1 =cv2.threshold(image,127,255,cv2.THRESH_BINARY)  # ret = return
cv2.imshow("1 Threshold Image", thresh1)

ret, thresh2 =cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)  # ret = return
cv2.imshow("2 Threshold Image", thresh2)

ret, thresh3 =cv2.threshold(image,127,255,cv2.THRESH_TRUNC)  # ret = return
cv2.imshow("3 Threshold Image", thresh3)

ret, thresh4 =cv2.threshold(image,127,255,cv2.THRESH_TOZERO)  # ret = return
cv2.imshow("4 Threshold Image", thresh4)

ret, thresh5 =cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)  # ret = return
cv2.imshow("5 Threshold Image", thresh5)


# In[29]:


image =cv2.imread('Hillary.jpg')
cv2.imshow('Original Image', image)

# Values below 127 goes to 0 and above 127 goes to 255
ret, thresh1 =cv2.threshold(image,127,255,cv2.THRESH_BINARY)  # ret = return
cv2.imshow("1 Threshold Image", thresh1)

ret, thresh2 =cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)  # ret = return
cv2.imshow("2 Threshold Image", thresh2)

ret, thresh3 =cv2.threshold(image,127,255,cv2.THRESH_TRUNC)  # ret = return
cv2.imshow("3 Threshold Image", thresh3)

ret, thresh4 =cv2.threshold(image,127,255,cv2.THRESH_TOZERO)  # ret = return
cv2.imshow("4 Threshold Image", thresh4)

ret, thresh5 =cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)  # ret = return
cv2.imshow("5 Threshold Image", thresh5)


# # AdaptiveThreshold Algorith

# In[30]:


image =cv2.imread('Origin_of_Species.jpg',0)
cv2.imshow('Original Image', image)
cv2.waitKey()

# it's good practice to blur image as it removes noise
image =cv2.GaussianBlur(image,(3,3),0)
#using AdaptiveThreshold Algorithm
thresh =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
cv2.imshow('adaptive mean threshold image', thresh)
cv2.waitKey()
cv2.destroyAllWindows()


# In[31]:


image =cv2.imread('Origin_of_Species.jpg',0)
cv2.imshow('Original Image', image)
cv2.waitKey()

# it's good practice to blur image as it removes noise
image =cv2.GaussianBlur(image,(3,3),0)
#using AdaptiveThreshold Algorithm
thresh =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
cv2.imshow('adaptive mean threshold image', thresh)
cv2.waitKey()

_,th2 =cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu threshold', th2)
cv2.waitKey()

blur =cv2.GaussianBlur(image,(5,5),0)
_,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu gaussian threshold',th3)
cv2.waitKey()

cv2.destroyAllWindows()


# In[32]:


image = cv2.imread('opencv_inv.png',0)
cv2.imshow("Original Image",image)
cv2.waitKey()

# Let's define our kernel size
kernel = np.ones((5,5))

#Erosion-remove pixel boundaries of image
#Dilation -Adds pixel bounaries of image
#Opening -Erosion  followed by dilation
#Closing - Dilation followed by erosion

#erosion
erosion =cv2.erode(image,kernel,iterations =1)
cv2.imshow("erosion Image", erosion)
cv2.waitKey()

#dialation
dilation =cv2.dilate(image,kernel,iterations =1)
cv2.imshow("dilation Image",dilation)
cv2.waitKey()

#Opening -good for removing noise
opening =cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
cv2.imshow("Opening Image", opening)
cv2.waitKey()

#closing -good for removing noise
closing =cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
cv2.imshow("closing Image", closing)
cv2.waitKey()

cv2.destroyAllWindows()


# # Edge Detection & Image Gradients

# In[33]:


image = cv2.imread('input.jpg',0)

height, width = image.shape

sobel_x =cv2.Sobel(image, cv2.CV_64F,0,1,ksize =5)
sobel_y =cv2.Sobel(image,cv2.CV_64F,1,0,ksize =5)

cv2.imshow("Original Image",image)
cv2.waitKey()
cv2.imshow("Sobal X Image",sobel_x)
cv2.waitKey()
cv2.imshow("Sobal Y Image",sobel_y)
cv2.waitKey()

cv2.destroyAllWindows()

# After using this code lot of disturbance,so better one is Canny


# In[34]:


image = cv2.imread('input.jpg',0)

height, width = image.shape

sobel_x =cv2.Sobel(image, cv2.CV_64F,0,1,ksize =5)
sobel_y =cv2.Sobel(image,cv2.CV_64F,1,0,ksize =5)

cv2.imshow("Original Image",image)
cv2.waitKey()
cv2.imshow("Sobal X Image",sobel_x)
cv2.waitKey()
cv2.imshow("Sobal Y Image",sobel_y)
cv2.waitKey()

sobel_or =cv2.bitwise_or(sobel_x,sobel_y)

cv2.imshow("Sobal X OR Y Image", sobel_or)
cv2.waitKey()

# LAPLACIAN

laplacian =cv2.Laplacian(image,cv2.CV_64F)
cv2.imshow("laplacian Image", laplacian)
cv2.waitKey()

#Canny Edge Detection -Best Result
canny = cv2.Canny(image,50,120)
cv2.imshow("canny Image",canny)
cv2.waitKey()

cv2.destroyAllWindows()


# In[ ]:



        


# In[35]:


os.getcwd()


# In[36]:


os.chdir('D:\\Learnbay\\Computer Vision\\Haarcascades')


# # Object Detection

# ## Face Detection

# In[37]:


import os
import numpy as np
import pandas as pd
import cv2

face_classifier =cv2.CascadeClassifier('D:\Learnbay\Computer Vision\Haarcascades\haarcascade_frontalface_default.xml')

image =cv2.imread('salman-khan.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces =face_classifier.detectMultiScale(gray,1.3,5)

if len(faces)==0:  #() =blank
    print("No face is found")
    
# Boundary Box    
for (x,y,w,h) in faces:
    cv2.rectangle(gray, (x,y),(x+w,y+h),(25,50,63),2)
    cv2.imshow("Face Detection",gray)
    cv2.waitKey()

cv2.destroyAllWindows()


# In[38]:


face_classifier =cv2.CascadeClassifier('D:\Learnbay\Computer Vision\Haarcascades\haarcascade_frontalface_default.xml')

image =cv2.imread('multiple_faces.jpg')
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces =face_classifier.detectMultiScale(image,1.3,5)

if len(faces)==0:  #() =blank
    print("No face is found")
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Face Detection",image)
    cv2.waitKey()

cv2.destroyAllWindows()


# # Eye Detection

# In[ ]:


#Face Detection
face_classifier =cv2.CascadeClassifier('D:\Learnbay\Computer Vision\Haarcascades\haarcascade_frontalface_default.xml')

#Eye Detection
eye_classifier =cv2.CascadeClassifier('D:\Learnbay\Computer Vision\Haarcascades\haarcascade_eye.xml')

image =cv2.imread('multiple_faces.jpg')
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces =face_classifier.detectMultiScale(image,1.3,5)

if len(faces)==0:  
    print("No face is found")
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Face Detection",image)
    cv2.waitKey()
    
    #eye_image = image[y:y+h, x:x+w]
    eye_color = image[y:y+h, x:x+w]
    eyes =eye_classifier.detectMultiScale(eye_color)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(eye_color, (ex,ey), (ex+ew,ey+eh),(255,0,0),3)
        cv2.imshow("Eye",image)
        cv2.waitKey()

cv2.destroyAllWindows()


# # Car Detection in Video

# In[ ]:


car_classifier =cv2.CascadeClassifier('D:\Learnbay\Computer Vision\Haarcascades\haarcascade_car.xml')
cap = cv2.VideoCapture('cars.avi') # cv2.VideoCapture for videos, cars.avi is file

while cap.isOpened():
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.imshow("Car is running on the highway",frame)
        
    if cv2.waitKey(1) ==13:  # to exit from the video unlike restart
        break
cap.relese()
cv2.destroyAllWindows()
        


# In[ ]:


25-


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





