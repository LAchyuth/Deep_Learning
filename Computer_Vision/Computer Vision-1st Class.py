#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install opencv-python


# In[2]:


import cv2
import os
import numpy as np


# In[3]:


os.getcwd()


# In[4]:


os.chdir('D:\\Learnbay\\Computer Vision\\images')


# In[5]:


input_img =cv2.imread('Modi.jpg')


# In[6]:


# View an image
cv2.imshow("Good Morning Modi Ji", input_img)
cv2.waitKey() # this allow the image to open
cv2.destroyAllWindows() # this will close the opened image
  


# In[7]:


input_img


# In[8]:


input_img.shape


# In[9]:


# Let's print each dimension of the input_img
print("Height of the image:", int(input_img.shape[0]),'pixels')
print("Width of the image:", int(input_img.shape[1]), 'pixels')
print("Channel of the image:", int(input_img.shape[2]),'RGB')


# In[10]:


# How to save the image(now new modi ji image saved with Modiji_jai_Bharat name)

cv2.imwrite("Modiji_Jai_Bharat.jpg", input_img)


# In[11]:


# Trying with new photo

input_img2 =cv2.imread("Obama.jpg")


# In[12]:


cv2.imshow("Good Morning Obamaji", input_img2)
cv2.waitKey()
cv2.destroyAllWindows()

# after image open, just keep cursor on the image and press enter to close the image


# In[13]:


input_img2


# In[14]:


cv2.imwrite("President Obama ji.png",input_img2)


# ## Convert Color image to Grayscale

# In[15]:


# Load an image
image = cv2.imread("Modi.jpg")

# View the image
cv2.imshow("Original Image", image)
cv2.waitKey()

# we use cvtColor, to convert to grayscale
gray_image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale image", gray_image)
cv2.waitKey()

color_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imshow("COLOR image", color_image)
cv2.waitKey()

cv2.destroyAllWindows()


# In[16]:


from PIL import Image # PIL= Python image library


# In[17]:


pic = Image.open('Jeffrey Hinton.jpg')
pic


# In[18]:


type(pic)


# In[19]:


pic_arr = np.asarray(pic)
pic_arr.shape


# In[20]:


import matplotlib.pyplot as plt
plt.imshow(pic_arr)
plt.show()


# In[21]:


pic_arr1 = pic_arr.copy()
pic_arr2 = pic_arr.copy() 
pic_arr3 = pic_arr.copy()
plt.imshow(pic_arr3)


# In[22]:


type(pic_arr1)


# In[23]:


pic_arr1


# In[24]:


# R -Red
# G -Green
# B - Blue
# 0 - Gray
# 1 - RGB
pic_arr1[:,:,0] # converted rgb to gray image


# In[25]:


plt.imshow(pic_arr1[:,:,0])


# In[26]:


plt.imshow(pic_arr1[:,:,1])


# In[27]:


plt.imshow(pic_arr1[:,:,2])


# In[28]:


pic_arr1[:,:,0] = 0 # target is red : red color will vanish
pic_arr2[:,:,1] = 0 # target is green : green color will vanish
pic_arr3[:,:,2] = 0 # target is blue : blue color will vanish

plt.imshow(pic_arr1)


# In[29]:


plt.imshow(pic_arr2) 


# In[30]:


plt.imshow(pic_arr3)


# In[31]:


# ComputerVision : CV2

# 0-grayscale and 1-rgb scale

image = cv2.imread('Trump.jpg',0)
image


# In[32]:


image1 = cv2.imread('Trump.jpg',1)
image1


# In[33]:


cv2.imshow("Trump Image",image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[34]:


cv2.imshow("Trump Image",image1)
cv2.waitKey()
cv2.destroyAllWindows()


# In[35]:


rgb_image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imshow("Trump Image",rgb_image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[36]:


rgb_image =cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
cv2.imshow("Trump Image",rgb_image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[37]:


# Capture all colors at one go

color_list =[method for method in dir(cv2) if method.startswith('COLOR_') is True]
color_list
len(color_list)

img = cv2.imread('opencv.jpg')

for color in range(len(color_list)):
    col_name = color_list[color]
    col_code = 'cv2.'+col_name
    resized = cv2.resize(img, (400,400))
    try:
        rgb_image = cv2.cvtColor(resized,eval(col_code))
        cv2.imshow(col_name + "_"+str(color),rgb_image)
        cv2.waitKey()
    except Exception as e:
        print("Error on :", str(col_code))
        pass
    cv2.destroyAllWindows()


# ## Drawing images and shapes using Open CV/CV2

# In[38]:


import numpy as np
import cv2

# Create a black image
image = np.zeros((512,512,3))  # color image 
image1 = np.zeros((512,512))   # black & white

cv2.imshow("Black rectangle (B&W)", image1)
cv2.waitKey(0)
cv2.imshow("Black rectangle (Color)",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[39]:


# Let's draw a line over black square

image2 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.line(image2,(0,0),(100,100),(2,255,255),5)
# pt1 =(0,0) orgin, pt2 =(100,100), 2,255,255 = represents intensity of blue,green,red color,5=thickness
cv2.imshow("Color Line", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[40]:


# Let's draw a rectangle over black square

image3 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image3,(100,100),(300,200),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[41]:


# Let's draw a square over black square

image4 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image4,(100,100),(200,200),(210,0,85),2)
# here -25 represents filling the box
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[42]:


# Let's draw a circle over black square

image5 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.circle(image5,(200,300),(100),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[43]:


image6 =np.zeros((512,512,3))
cv2.putText(image6, "Hello",(25,300),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image6)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[47]:


# All above images one by one

image2 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.line(image2,(0,0),(100,100),(2,255,255),5)
# pt1 =(0,0) orgin, pt2 =(100,100), 2,255,255 = represents intensity of blue,green,red color,5=thickness
cv2.imshow("Color Line", image2)
cv2.waitKey(0)

image3 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image3,(100,100),(300,200),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image3)
cv2.waitKey(0)

image4 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image4,(100,100),(200,200),(210,0,85),2)
# here -25 represents filling the box
cv2.imshow("Rectangle", image4)
cv2.waitKey(0)

image5 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.circle(image5,(200,300),(100),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image5)
cv2.waitKey(0)

image6 =np.zeros((512,512,3))
cv2.putText(image6, "Hello",(25,300),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image6)
cv2.waitKey(0)

image6 =np.zeros((512,512,3))
cv2.putText(image6, "Hello",(25,300),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image6)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[50]:


# All above images one by one

image2 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.line(image2,(0,0),(100,100),(2,255,255),5)
# pt1 =(0,0) orgin, pt2 =(100,100), 2,255,255 = represents intensity of blue,green,red color,5=thickness
cv2.imshow("Color Line", image2)
cv2.waitKey(0)

image3 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image3,(100,100),(300,200),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image3)
cv2.waitKey(0)

image4 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.rectangle(image4,(100,100),(200,200),(210,0,85),2)
# here -25 represents filling the box
cv2.imshow("Rectangle", image4)
cv2.waitKey(0)

image5 = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.circle(image5,(200,300),(100),(210,0,85),-25)
# here -25 represents filling the box
cv2.imshow("Rectangle", image5)
cv2.waitKey(0)

image6 =np.zeros((512,512,3))
cv2.putText(image6, "Hello",(25,300),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image6)
cv2.waitKey(0)

image7 =np.zeros((512,512,3))
cv2.putText(image7, "Achyuth",(25,300),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image7)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[67]:


# All images at once

image = np.zeros((512,512,3)) # maximum size(512W x 512H & 3-color image)
cv2.line(image,(0,0),(500,500),(2,255,255),5)
# pt1 =(0,0) orgin, pt2 =(100,100), 2,255,255 = represents intensity of blue,green,red color,5=thickness

cv2.rectangle(image,(300,100),(500,200),(210,0,85),-25)
# here -25 represents filling the box

cv2.rectangle(image,(150,150),(250,250),(50,0,0),-25)
# here -25 represents filling the box

cv2.rectangle(image,(100,100),(200,200),(210,0,85),2)
# here -25 represents filling the box

cv2.circle(image,(400,400),(100),(210,0,85),-25)
# here -25 represents filling the box

cv2.putText(image, "Hello",(25,400),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,2),3)

cv2.putText(image, "Achyuth",(25,300),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(0,0,2),3)
cv2.imshow("Hello World!", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




