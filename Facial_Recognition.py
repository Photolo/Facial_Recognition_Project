import cv2
import numpy as np
import face_recognition
import os

path = 'faces'

images = []
classNames = []

# Get IMages
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)

# Change Image Size, Lower better quality, higher slower.

scale = 0.25
box_multiplier = 1/scale


# Find Encodings
def finEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)
    return encodeList

# Find encodings of training images

knownEncodes = finEncodings(images)
print('Encoding Complete')
 
# Define a videocapture object
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()  # Reading Each frame

    # Change Size
    Current_image = cv2.resize(img,(0,0),None,scale,scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    # Face location and encodings for current frame

    face_locations = face_recognition.face_locations(Current_image, model='cnn')
    face_encodes = face_recognition.face_encodings(Current_image,face_locations)
# Find the matches

    for encodeFace,faceLocation in zip(face_encodes,face_locations):
        matches = face_recognition.compare_faces(knownEncodes,encodeFace, tolerance=0.6)
        # print(matches)
        faceDis = face_recognition.face_distance(knownEncodes,encodeFace)
        matchIndex = np.argmin(faceDis)

        # If match found then get the class name for match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

        else:
            name = 'Unknown'

        y1,x2,y2,x1 = faceLocation
        y1, x2, y2, x1 = int(y1box_multiplier),int(x2box_multiplier),int(y2box_multiplier),int(x1box_multiplier)

        # Draw rectangle around detected face

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-20),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,0.5,(255,255,255),2)


    # sOutput
    cv2.imshow('Webcam',img)

    if cv2.waitKey(1) == ord('q'):
        break

# End

cap.release()
cv2.destroyAllWindows()
