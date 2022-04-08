#This is a program to detect eyes and face
import cv2

capture = cv2.VideoCapture(0)

face_c = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_c = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_c.detectMultiScale(gray, 1.3, 5)   #shrink

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]	#region of interest
        roi_color = frame[y:y+w, x:x+w]
        eyes = eye_c.detectMultiScale(roi_gray, 1.3, 5)    #detect eyes from the face
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('a'):
        break

capture.release()
cv2.destroyAllWindows()