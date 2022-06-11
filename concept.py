#Project by Dayton Berezoski
#Contact Email: daytonbere@gmail.com
#Created on June 4, 2022

#Retired on June 10, 2022 moving application over to javascript
#Kept as proof of concept

import cv2 as cv
import mediapipe as mp
import math

mpface_mesh = mp.solutions.face_mesh
face_mesh = mpface_mesh.FaceMesh(max_num_faces = 1)

lip_points = {
    0 : (0, 0),
    12 : (0, 0),
    15 : (0, 0),
    17 : (0, 0),
    39 : (0, 0),
    57 : (0, 0),
    181 : (0, 0),
    269 : (0, 0),
    287 : (0, 0),
    405 : (0, 0)
}

alignment_points = {
    #vertical
    1 : (0, 0),
    6 : (0, 0),
    #horizontal
    33 : (0, 0),
    263 : (0, 0)
}

#lip_corres_points[top_point] = bottom_point
lip_corres_points = {
    0 : 17,
    12 : 15,
    39 : 181,
    269 : 405
}

#alignment_corres_points[top_point] = bottom_point
alignment_corres_points = {
    1 : 6,
    33 : 263
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
    return (int((top[0]+bottom[0]) / 2), int((top[1]+bottom[1]) / 2))

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
                    lip_points[id] = (x, y)
                
                if id in alignment_points:
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)   
                    alignment_points[id] = (x, y)
    
    #Place a dot for every lip articulation point
    for id in lip_points:
        frame = cv.circle(frame, lip_points[id], 1, (255,255,0), 4)

    #Place a different dot for every alignment point
    for id in alignment_points:
        frame = cv.circle(frame, alignment_points[id], 1, (12,255,0), 4)

    for id in lip_corres_points:
        top = lip_points[id]
        bottom = lip_points[lip_corres_points[id]]
        distance = length(top, bottom)
        mid = midpoint(top, bottom)
        frame = cv.putText(frame, str(distance) + "px", pos[id], cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
        frame = cv.circle(frame, mid, 1, (255,0,255), 4)

    for id in alignment_corres_points:
        start = alignment_points[id]
        end = alignment_points[alignment_corres_points[id]]
        frame = cv.line(frame, start, end, (0, 0, 255), 4)

    rotation = alignment_points[1][0]-alignment_points[6][0]
    if rotation > 5:
        frame = cv.putText(frame, "Rotate clockwise", (1200,300), cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
    elif rotation < -5:
        frame = cv.putText(frame, "Rotate counter-clockwise", (1200,300), cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
    else:
        tilt  = alignment_points[6][1] - int((alignment_points[33][1] + alignment_points[263][1]) / 2)
        if tilt > 13:
             frame = cv.putText(frame, "Tilt head up", (1200,300), cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
        elif tilt < -5:
            frame = cv.putText(frame, "Tilt head down", (1200,300), cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)
        else:
            frame = cv.putText(frame, "Keep head still", (1200,300), cv.FONT_HERSHEY_TRIPLEX, 1, (255,0,255), 2)

    cv.imshow('test face mesh', frame)
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()