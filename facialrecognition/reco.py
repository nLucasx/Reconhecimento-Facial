import cv2
import pickle

model_path = 'data/haarcascade_frontalface_alt2.xml'

cascade = cv2.CascadeClassifier(model_path)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')

dictionary = open('labels.pickle', 'rb')
original_labels = pickle.load(dictionary)
reversed_labels = {v:k for k, v in original_labels.items()}

cap = cv2.VideoCapture(0)

while cv2.waitKey(5) != 113:

    (ret, frame) = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, width, height) in faces:
        end_x = x + width
        end_y = y + height
        stroke = 2
        region = gray[y:end_y, x:end_x]
        colored_region = frame[y:end_y, x:end_x]
        label_id, confidence = face_recognizer.predict(region)

        if confidence >= 5 and confidence <= 75:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = reversed_labels[label_id]
            color = (255, 0, 0)
            cv2.putText(frame, name, (x, y-20), font, 1, color, stroke, cv2.LINE_AA)
        
        color = (0, 0, 255)
        cv2.rectangle(frame, (x,y), (end_x, end_y), color, stroke)
    
    cv2.imshow('Recognition', frame)