import cv2
import numpy as np
import os
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode

def put_text(frame,text,x,y):
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),2)

# Define a function to load the training data
def load_training_data(data_dir):
    # Create an empty list to store the training data
    faces = []
    labels = []
    
    # Loop through each subdirectory in the data directory
    for sub_dir in os.listdir(data_dir):
        sub_dir_path = os.path.join(data_dir, sub_dir)
        
        # Loop through each image file in the subdirectory
        for filename in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, filename)
            
            # Load the image file as a grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert the grayscale image to a numpy array and append it to the faces list
            faces.append(np.array(img, dtype=np.uint8))
            
            # Append the label (subdirectory name) to the labels list
            labels.append(int(sub_dir))
    
    # Return the faces and labels lists as numpy arrays
    return np.array(faces), np.array(labels)

# Define the paths to the training data directory and the output YAML file
data_dir = 'training_data'
output_file = 'training_data.yml'

# Load the training data
faces, labels = load_training_data(data_dir)

# Train the facial recognition model using the loaded training data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Save the trained model to a YAML file
recognizer.write(output_file)

# Load the trained model from the YAML file
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(output_file)
name = {0: "name1", 1: "name2", 2: "name3"}

# Define the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the video capture device
video_capture = cv2.VideoCapture(0)

# Loop over frames from the video capture device
while True:
    # Capture the current frame from the video capture device
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        
        # Extract the face region from the grayscale frame
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # Recognize the face using the trained model
        label, confidence = recognizer.predict(face_roi_gray)
        
        # Print the predicted label and confidence score on the frame
        cv2.putText(frame, f'Label: {label}, Confidence: {confidence:.1f}', (x, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        predict_name=name[label]

        if confidence>55:
            t=1
            put_text(frame,predict_name,x,y) 
            try:
                connection = mysql.connector.connect(host='localhost',
                                                     database='newdb',
                                                     user='root',
                                                     password='password')
                if predict_name == 'name1':
                    mySql_insert_query = "INSERT INTO database1 VALUES (1, 'name1')"
                if predict_name == 'name2':
                    mySql_insert_query = "INSERT INTO database1 VALUES (2, 'name2')"
                if predict_name == 'name3':
                    mySql_insert_query = "INSERT INTO database1 VALUES (3, 'name3')"

                cursor = connection.cursor()
                cursor.execute(mySql_insert_query)
                connection.commit()
                print(cursor.rowcount, "Record inserted successfully into Data table")
                cursor.close()

            except mysql.connector.Error as error:
                print("Failed to insert record into Data table {}".format(error))
                break

            finally:
                if (connection.is_connected()):
                    connection.close()
                    print("MySQL connection is closed")
                    break
            
    # Display the resulting frame
    cv2.imshow('Video', frame)
        
    # Exit the loop if the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
