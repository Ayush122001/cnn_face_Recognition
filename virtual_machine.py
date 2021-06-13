import cv2
import numpy as np
from keras.models import load_model
import os
model = load_model('ayush_recognition_final.h5')
m = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
flag = 0
cap = cv2.VideoCapture(0)
while True:    
    rc, photo = cap.read()
    face = m.detectMultiScale(photo)
    if len(face) != 0:
        x1 = face[0][0]
        y1 = face[0][1]
        x2 = x1 + face[0][2]
        y2 = y1 + face[0][3]
        frame = cv2.rectangle(photo,(x1,y1),(x2,y2),[255,0,0],4)
        final_photo = photo[y1:y2,x1:x2]
        test = cv2.resize(final_photo,(64,64))
        final_test = np.expand_dims(test, axis=0)
        if model.predict(final_test)[0][0] < 0.05:
            cv2.putText(photo, "Welcome Ayush", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            count = count + 1
            cv2.imshow('Validating',photo)
        else:
            cv2.imshow('Validating',photo)
        if count == 15 and model.predict(final_test)[0][0] < 0.05:
            count = 16
            os.system('VBoxManage startvm "docker testing"')
    else:
        cv2.imshow('Validating',photo)
    if cv2.waitKey(1) == 13:
        break
        
cv2.destroyAllWindows()
cap.release()
