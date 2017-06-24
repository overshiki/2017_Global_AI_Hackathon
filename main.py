import time

import cv2

vidcap = cv2.VideoCapture('data/test1.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    count += 1
    if count % 50 != 0:
        continue
    success,image = vidcap.read()
    print 'Read a new frame: ', success, count

    # Create the haar cascade
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(2)
