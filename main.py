#Project by Dayton Berezoski
#Contact Email: daytonbere@gmail.com
#Created on June 4, 2022

import cv2 as cv
import mediapipe as mp

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
mp_draw = mp.solutions.drawing_utils
mpface_mesh = mp.solutions.face_mesh
face_mesh = mpface_mesh.FaceMesh(max_num_faces = 1)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 0))

lip_points = {
    0 : [0, 0],
    12 : [0, 0],
    15 : [0, 0],
    17 : [0, 0],
    39 : [0, 0],
    57 : [0, 0],
    181 : [0, 0],
    269 : [0, 0],
    287 : [0, 0],
    405 : [0, 0]
}

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

    #Gets the points of the face mesh but does not display the face mesh
    if results.multi_face_landmarks: 
        for face_lms in results.multi_face_landmarks:
            for id, lm in enumerate(face_lms.landmark):
                if id in lip_points:
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)   
                    lip_points[id] = [x, y]
    
    #Place a dot for every lip articulation point
    for id in lip_points:
        frame = cv.circle(frame, (lip_points[id][0], lip_points[id][1]), 1, (255,255,0), 4)
    
    cv.imshow('test face mesh', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()