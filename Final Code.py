#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance 
import vlc
import numpy as np


# In[3]:


alerty=vlc.MediaPlayer('Sounds/yawn.mp3')
alerte=vlc.MediaPlayer('Sounds/SLEEP.mp3')
alertt=vlc.MediaPlayer('Sounds/TAKE REST.mp3')
thresh=0.25
thresh2=0.6
flag=0
flag2=0
flag3=0


# In[4]:


def getFaceDirection(shape, size):
    image_points = np.array([
                            (359, 391),     # Nose tip
                            (399, 561),     # Chin
                            (337, 297),     # Left eye left corner
                            (513, 301),     # Right eye right corne
                            (345, 465),     # Left Mouth corner
                            (453, 469)      # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])


# In[5]:


def eye_aspect_ratio(eye) :
        A=distance.euclidean(eye[1],eye[5])
        B=distance.euclidean(eye[2],eye[4])
        C=distance.euclidean(eye[0],eye[3])
        ear = (A+B)/(2.0*C)
        return ear


# In[6]:


def mouth_aspect_ratio(mouth) :
    Am=distance.euclidean(mouth[2],mouth[10])
    Bm=distance.euclidean(mouth[4],mouth[8])
    Cm=distance.euclidean(mouth[0],mouth[6])
    mar = (Am+Bm)/(2.0*Cm)
    return mar


# In[7]:


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]


# In[8]:


detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
#predict2=dlib.shape_predictor("Models/shape_predictor_81_face_landmarks.dat")


# In[9]:


frame_check=20
frame_check2=10
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    size = frame.shape
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subjects = detect(gray,0)
    for subject in subjects :
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        #shape2=predict2(gray, subject)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        lips=shape[mStart:mEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        mEAR = mouth_aspect_ratio(lips)
        ear = (leftEAR + rightEAR)/2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(lips)
        
        cv2.drawContours(frame, [leftEyeHull],-1, (255,255,255), 1)
        cv2.drawContours(frame, [rightEyeHull],-1, (255,255,255), 1)
        cv2.drawContours(frame, [mouthHull],-1, (255,255,255), 1)
        if mEAR>thresh2 :
            flag2+=1
            #print("YAWN",flag2)
            cv2.putText(frame, "YAWNING!!!", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
            cv2.putText(frame, "YAWNING!!!", (10,325),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            alerty.play()

        if ear<thresh :
            flag+=1
            #print(flag)
            if flag>=frame_check:
                cv2.putText(frame, "DROWSINESS DETECTED!!!", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
                cv2.putText(frame, "FEELING SLEEPY!!!", (150,325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                alertt.play()
        elif(ear>thresh):
            flag=0
            flag2=0
        if(ear>thresh and mEAR<thresh2):
            alertt.stop()
            alerty.stop()
          
        #head position  
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        count=0 
        if(flag>=frame_check and getFaceDirection(shape, size)<0):
            count+=1
            cv2.putText(frame, "SIT STRAIGHT!!!", (10, 225),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
           


    cv2.imshow('Frame',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



