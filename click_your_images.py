import cv2
cap = cv2.VideoCapture(1)
cap.release()
count = 0
while count < 15000:
    rc, photo = cap.read()
    model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = model.detectMultiScale(photo)
    if len(face) != 0:
        x1 = face[0][0]
        y1 = face[0][1]
        x2 = x1 + face[0][2]
        y2 = y1 + face[0][3]
        frame = cv2.rectangle(photo,(x1,y1),(x2,y2),[255,0,0],4)
        final_photo = photo[y1:y2,x1:x2]
        test = cv2.resize(final_photo,(170,170))
        cv2.imshow('hi',test)
        cv2.waitKey(1)
        path = 'ds/Training/ayush/'+str(count)+'.jpg' 
        cv2.imwrite(path,test)
        count = count + 1
        print(count)
    else:
        pass
cv2.destroyAllWindows()
cap.release()
