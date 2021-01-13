import cv2
import dlib
import sys
from deepface import DeepFace

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('Cannot open camera')

actions = ['emotion']

while cap.isOpened():
    ret, frame = cap.read()
    predictions = DeepFace.analyze(frame, enforce_detection=False, actions=actions, detector_backend='ssd')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) != 1:
        print('There are more than one face in your image or there is no face. Please choose an image that contains one face')
        sys.exit(0)

    else:
        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
            cv2.putText(frame, predictions['dominant_emotion'], (50, 50),  cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
