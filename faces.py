#4eme
from cProfile import label
from cv2 import rectangle
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
#we add our trainned data
recognizer.read("./recognizers/face-trainner.yml")

#extraire le label du fichier pickel
labels={"person_name":1}
with open("./pickles/face-labels.pickle",'rb') as f :
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()   

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
		# recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            cv2.putText(frame,labels[id_],(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,0,255),2,cv2.LINE_AA)
			#put the name in the image
			#print(labels[id_])
			#print(id_) 

        img_item ="my-image.png"

        #dessine carre sur visage 
        color = (255, 0, 0) #BGR 0-255 
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, 2)

		# subitems = smile_cascade.detectMultiScale(roi_gray)
    	# for (ex,ey,ew,eh) in subitems:
    	# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	
    # Display the resulting frame
    cv2.imshow('coucou les amis',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()