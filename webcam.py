#!usr/env/python3

''' python script that will periodically print true or false if it can detect a face on the webcams feed '''

import cv2
import time
import numpy as np

class Camera(object):
    def __init__(self, source=0, cascPath='haarcascade_frontalface_default.xml'):
        self.cascPath = cascPath
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.video_capture = cv2.VideoCapture(0)

        self.last_frame = np.zeros((480, 640, 3))
        self.movement_zone = np.zeros((480,640,3))

        self.bool_face_detected = False
        self.num_faces_detected = 0

    def exit(self):
        # When everything is done, release the capture
        self.video_capture.release()

    def update(self):
        ret, frame = self.video_capture.read()
        self.detect_movement(frame)
        self.last_frame = frame # 480 x 640 X 3 array

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            self.bool_face_detected = True
            self.num_faces_detected = len(faces)
        else:
            self.bool_face_detected = False
            self.num_faces_detected = 0
    
    def detect_movement(self, frame):
        ''' given the current frame, and the self.last_frame, 
        find if there is any movement based on a threshold'''
        try:
            if self.last_frame == None:
                return
        except:
            return

        self.movement_zone = np.array(np.shape(self.last_frame))
        for i in range(480):
            for j in range(640):
                for k in range(3):
                    self.movement_zone[i][j][k] = self.last_frame[i][j][k] - frame[i][j][k]
        return

if __name__ == "__main__":
    cam = Camera()
    while True:
        st = time.monotonic()
        cam.update()
        et = time.monotonic()
        t = float((et - st) * 1000)
        fps = round(float(1000 / t))


        print(f"FPS= {fps} || face?: {cam.bool_face_detected} - num_faces: {cam.num_faces_detected}")
        # try:
        #     print(cam.movement_zone)
        # except:
        #     pass