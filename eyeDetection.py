import numpy as np
import cv2
import time

# This is the haar cascade classifier for face detection
# It is going to be used to find the boxes around faces in an image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# This is the haar cascade classifier for eye detection
# This is used for p
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# U can think of the cap object  as the camera 
cap = cv2.VideoCapture(0)


while True:
    #Entering an infinite loop

    #Reading an image from the camera (BGR image is stored in img variable)
    ret, img = cap.read()

    #Converting the image to grayscale    (Gray Scale image is stored in gray variable)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Collecting the faces in the faces array
    # There might be multiple faces in an image
    # Hence this function returns an array of faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
    # Getting the co-ordinates (face box) of each face at a time from the faces array

        # Drawing a rectangle in the BGR imgage stored in the img variable
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # this is the face region of the gray scale image
        roi_gray = gray[y:y+h, x:x+w]
        # this is the face region of the colored image
        roi_color = img[y:y+h, x:x+w]
        
        # USing the haar cascades for collecting the eyes in the face region
        # Observe that haar cascade is being used on the gray scale image
        # THe colored image is mainly being kept for displaying purpose
        # the parameters in the detectMultiscale function are:
        # The grayscale image of the face region
        # Scaling factor and minNeighbours - IF u forgot what these are just do  a quick google search
        # U may need to tweak the parameter values a bit at times for adjustment
        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=10)
        
        # This will be used to store the images of the two eyes
        eyePair=[]
        for (ex,ey,ew,eh) in eyes:
        # Collecting the eyes one at a time
        # Since we used the haar cascade on the face region
        # The above loop should be running twice
        # First for one eye and second for another eye
        # 
            # Drawing the eyes on the image "roi_color"
            # "roi_color" is a part of "img"    
            # Observe that drawing on the "roi_color" will result in drawing on the "img"
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #Collecting the eye images in the eyePair array    
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            eyePair.append(eye)
        
        
        
        if(len(eyePair)==2):
        # checking if both eyes have been collected or not

            #Making the time stamp       
            timeStamp=time.time() 
            
            #keeping it upto seconds only
            timeStamp=int(timeStamp)
            timeStamp=str(timeStamp)
            #print(timeStamp)

            #Now the code for saving the images in the eyes directory begins
            #cv2.imwrite(filename, img)
            
            #Saving the first eye with time stamp
            eye1Name="eye"+timeStamp+"Left.jpg"
            eye1=eyePair[0]
            cv2.imwrite('./eyes/'+eye1Name,eye1)

            #Saving the second eye with time stamp
            eye2Name="eye"+timeStamp+"Right.jpg"
            eye2=eyePair[1]
            cv2.imwrite('./eyes/'+eye2Name,eye2)

            
            # Ignore this for now, it is for seperation of the skin
            # grayEye1 = cv2.cvtColor(eye1,cv2.COLOR_BGR2GRAY)
            # ret1, thresh1 = cv2.threshold(grayEye1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # thresh1 =thresh1.reshape((thresh1.shape[0],thresh1.shape[1],1))

            # grayEye2 = cv2.cvtColor(eye2,cv2.COLOR_BGR2GRAY)
            # ret2, thresh2 = cv2.threshold(grayEye2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # thresh2 =thresh2.reshape((thresh2.shape[0],thresh2.shape[1],1))

            # result1=eye1*thresh1
            # result2=eye2*thresh2

            cv2.imshow('leftEye',eye1)
            cv2.imshow('righttEye',eye2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()