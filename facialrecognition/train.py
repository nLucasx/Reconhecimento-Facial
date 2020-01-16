import cv2
import numpy
import os
import pickle
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = 'data/haarcascade_frontalface_alt2.xml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists('images'):
    os.mkdir('images')

images_path = os.path.join(current_dir, 'images')
cascade = cv2.CascadeClassifier(model_path)

index_id = 0
label_ids = {}
face_regions = []
labels = []

for (root, dirs, files) in os.walk(images_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            
            file_path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "")
            
            if not label in label_ids:
                label_ids[label] = index_id
                index_id += 1

            current_id = label_ids[label]

            image = Image.open(file_path).convert('L')
            resolution = (450, 400)
            final_image = image.resize(resolution, Image.ANTIALIAS)
            
            pixel_matrix = numpy.array(final_image, 'uint8')

            faces = cascade.detectMultiScale(pixel_matrix, scaleFactor=1.5, minNeighbors=5)
            
            for (x, y, width, height) in faces:
                region = pixel_matrix[y:y+height, x:x+width]
                face_regions.append(region)
                labels.append(current_id)


new_dictionary = open('labels.pickle', 'wb')
pickle.dump(label_ids, new_dictionary)

face_recognizer.train(face_regions, numpy.array(labels))
face_recognizer.save('trained.yml')