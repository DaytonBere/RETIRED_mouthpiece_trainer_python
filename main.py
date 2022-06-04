#Project by Dayton Berezoski
#Contact Email: daytonbere@gmail.com
#Created on June 4, 2022

import cv2 as cv
import mediapipe as mp

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
mp_draw = mp.solutions.drawing_utils
mpface_mesh = mp.solutions.face_mesh
face_mesh = mpface_mesh.FaceMesh(max_num_faces = 2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_lms, mpface_mesh.FACE_CONNECTIONS, draw_spec, draw_spec)
            for id, lm in enumerate(face_lms.landmark):
                ih, iw, ic = frame.shape
                x, y = int(lm.x*iw), int(lm.y*ih)

    '''
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    '''

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()