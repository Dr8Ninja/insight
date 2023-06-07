import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np

model = insightface.app.FaceAnalysis(name='buffalo_m')
model.prepare(ctx_id=0, det_size=(640, 640))

dataset_folder = '/data/vishal/insight/face'


# image = cv2.imread('friends.jpg')
# image = image[:,:,::-1]

# image = ins_get_image('t1')

image_path = '/data/vishal/insight/output_frames/frame_110.jpg'
image = ins_get_image(image_path)

faces = model.get(image)

def calculate_similarity(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2)
    return similarity

for face in faces:
    for person_folder in os.listdir(dataset_folder):
        person_path = os.path.join(dataset_folder, person_folder)
        
        if os.path.isdir(person_path):  # Check if it's a directory
            for filename in os.listdir(person_path):
                image_path = os.path.join(person_path, filename)
                person_image = ins_get_image(image_path)
                
                person_faces = model.get(person_image)
                
                for person_face in person_faces:
                    similarity = calculate_similarity(face.embedding, person_face.embedding)
                    
                    similarity_threshold = 0.25
                    
                    if similarity > similarity_threshold:
                        face.name = person_folder
                
    box = face.bbox.astype(int)
    color = (0, 0, 255)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
    cv2.putText(image, face.name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


output_image_path = 'out_110.jpg'
cv2.imwrite(output_image_path, image)