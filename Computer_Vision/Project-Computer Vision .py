#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Small Project
# CNN -CV


# In[1]:


import os
import cv2


# In[12]:


os.getcwd()


# In[13]:


os.chdir('D:\Learnbay\Computer Vision')


# # Project 1: Body Movement Detection

# In[14]:


import numpy as np
import cv2


# In[15]:


body_classifier =cv2.CascadeClassifier("D:\\Learnbay\\Computer Vision\\Haarcascades\\haarcascade_fullbody.xml")

cap =cv2.VideoCapture("D:\\Learnbay\\Computer Vision\\images\\vtest.mp4")
while cap.isOpened():
    ret,frame =cap.read()
    
    if not ret:
        break  # close the video after it ends
            
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    for(x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Pedestrians",frame)
    if cv2.waitKey(1)==30:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# # Project 2:Video Count

# In[ ]:


import numpy as np
import cv2
from time import sleep

# Enable Web Camera
cap =cv2.VideoCapture("D:\\Learnbay\\Computer Vision\\images\\video.mp4")

while True:
    ret,frame =cap.read()
    cv2.imshow("Video Originals", frame)
    
    if cv2.waitKey(1)==25:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2
from time import sleep

# Enable Web Camera
cap =cv2.VideoCapture("D:\\Learnbay\\Computer Vision\\images\\video.mp4")
algo =cv2.createBackgroundSubtractorMOG2()  # frame by frame finding the image

while True:
    ret,frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting the color
    blur = cv2.GaussianBlur(gray, (3,3),5)          # image becomes blur
    
    # Applying on each frame
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))  # increase the thickness of the moving image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))  # change the moving object color to white
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow("Detector",dilatada)
    
    if cv2.waitKey(1)==25:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2
from time import sleep

min_width_rec = 80
min_height_rec = 80

offset = 6  # It will wait till 6sec, if no movement it through error
delay =60   # frame per second to video
carros = 0  #count of no of detected vehicles

count_lines_pos = 550
detec = []
def central_handle(x,y,w,h):  # calculate center coordinate of a rectangle
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x +x1
    cy = y + y1
    return cx, cy


# Enable Web Camera
cap =cv2.VideoCapture("D:\\Learnbay\\Computer Vision\\images\\video.mp4")
algo =cv2.createBackgroundSubtractorMOG2()  # frame by frame finding the image

# cv2.createBackgroundSubtractorMOG2() - this will consider movement in granular level
# cv2.bgsegm.createBackgroundSubtractorMOG() - this will not consider granular level movement

while True:
    ret,frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting the color
    blur = cv2.GaussianBlur(gray, (3,3),5)          # image becomes blur
    
    # Applying on each frame
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))  # increase the thickness of the moving image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))  # change the moving object color to white
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame,(25,count_lines_pos),(1200,count_lines_pos),(0,0,255),3)
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >=min_width_rec) and (h >=min_height_rec)
        
        if not validar_contorno:
            continue
            
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        #rectangle(img, pt1, pt2,color,thickness)
        cv2.putText(frame, "vehicle" + str(carros), (x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)
        #putText(img, text, org, fontFace, fontScale,color,thickness)
    
        center = central_handle(x,y,w,h)
        detec.append(center)
        cv2.circle(frame,center,4,(255,100,100),-1)
        #circle(img, center, radius, color,thickness)
    
    #Loop
    
    for(x,y) in detec:
        if y < (count_lines_pos + offset) and y >(count_lines_pos - offset):
            carros +=1
            cv2.line(frame,(25,count_lines_pos),(1200,count_lines_pos),(0,0,255),3)
            detec.remove((x,y))
            print("Vehicle is detected :" + str(carros))
            
    cv2.putText(frame, "Vehicle Count :" + str(carros),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),5)
    # putText(img, text, org, fontFace, fontScale, color, thickness)
    
    
    cv2.imshow("Original Video with vehicle count",frame)
    
    if cv2.waitKey(1)==25:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# # Project3 - Hand Gesture

# In[5]:


get_ipython().system('pip install mediapipe')


# In[7]:


pip install --upgrade mediapipe


# In[2]:


pip install --upgrade numpy


# In[2]:


import cv2
import numpy as np
import time
import mediapipe as mp


# In[ ]:


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
# Frame per second

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    imgBGR = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgBGR)
    print(results.multi_hand_landmarks)
    cv2.imshow("my image",img)
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                print(id,lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                
                if id ==4:
                    cv2.circle(img,(cx,cy), 15,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image",img)
        if cv2.waitKey(1)==13:
            break

cap.release()     
cv2.destroyAllWindows()        
        


# In[1]:


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
# Frame per second

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    imgBGR = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgBGR)
    print(results.multi_hand_landmarks)
    cv2.imshow("my image",img)
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                
                if id ==4:
                    cv2.circle(img,(cx,cy), 15,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image",img)
        if cv2.waitKey(1)==13:
            break

cap.release()     
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




