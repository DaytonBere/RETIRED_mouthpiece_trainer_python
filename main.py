#Project by Dayton Berezoski
#Contact Email: daytonbere@gmail.com
#Created on June 4, 2022

import cv2 as cv
import mediapipe as mp
import math

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

alignment_points = {
    #vertical
    1 : [0, 0],
    6 : [0, 0],
    #horizontal
    33 : [0, 0],
    263 : [0, 0]
}

#corres_points[top_point] = bottom_point
corres_points = {
    0 : 17,
    12 : 15,
    39 : 181,
    269 : 405
}

#positions for text of lengths
pos = {
    0 : (10, 30),
    12 : (10, 55),
    39 : (10, 80),
    269 : (10, 105)
}

#Calculates a truncated euclidean distance between the top and bottom points
def length(top, bottom):
    return int(math.sqrt(math.pow(top[0]-bottom[0], 2) + math.sqrt(math.pow(top[1]-bottom[1], 2))))

#Returns the midpoint between the top and bottom points
def midpoint(top, bottom):
    return [int((top[0]+bottom[0])//2), int((top[1]+bottom[1])//2)]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #Mirror the image
    frame = cv.flip(frame, 1)
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
                if id in alignment_points:
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)   
                    alignment_points[id] = [x, y]
    
    #Place a dot for every lip articulation point
    for id in lip_points:
        frame = cv.circle(frame, (lip_points[id][0], lip_points[id][1]), 1, (255,255,0), 4)

    #Place a different dot for every alignment point
    for id in alignment_points:
        frame = cv.circle(frame, (alignment_points[id][0], alignment_points[id][1]), 1, (12,255,0), 4)

    for id in corres_points:
        top = lip_points[id]
        bottom = lip_points[corres_points[id]]
        distance = length(top, bottom)
        mid = midpoint(top, bottom)
        frame = cv.putText(frame, str(distance) + "px", pos[id], cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
        frame = cv.circle(frame, (mid[0], mid[1]), 1, (255,0,255), 4)

    
    cv.imshow('test face mesh', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()