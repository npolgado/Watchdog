#!usr/env/python3

''' python script that will periodically print true or false if it can detect a face on the webcams feed '''

import cv2
import time
import numpy as np

class Camera(object):
    def __init__(self, source=0, cascPath='haarcascade_frontalface_default.xml', init_threshold=False):
        self.cascPath = cascPath
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.video_capture = cv2.VideoCapture(0)

        self.last_frame = np.zeros((480, 640))
        self.movement_zone = np.zeros((480, 640))
        self.movement_avg = 0
        self.movement_threshold = 0
        self.detected_movement = False

        self.bool_face_detected = False
        self.num_faces_detected = 0
        
        if init_threshold: self.movement_threshold = round(self.find_threshold(),2)
        else: self.movement_threshold = 65

    def exit(self):
        # When everything is done, release the capture
        self.video_capture.release()

    def update(self):
        ret, frame = self.video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.movement_zone = gray - self.last_frame
        self.movement_avg = round(np.average([np.average(i) for i in self.movement_zone]),2)
        # print(self.movement_avg)

        if self.movement_avg > self.movement_threshold: self.detected_movement = True
        else: self.detected_movement = False

        self.last_frame = gray # 480 x 640 X 3 array

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

    def find_threshold(self, bias=1):
        print("FINDING THRESHOLD FOR MOTION....")
        ans = 1000
        all = []
        st = time.monotonic()
        try:
            et = time.monotonic()
            dt = float(et - st)
            while dt <= 10:
                self.update()
                all.append(self.movement_avg)
                if self.movement_avg < ans: ans = self.movement_avg
                et = time.monotonic()
                dt = float(et - st)
                print(f"\r   {ans}   ", end="\r")
            if np.average(all) < ans:
                ans -= bias
            else:
                ans += bias
        except Exception as e:
            print(f"ERROR getting threshold\n{e}")
        print(f"FOUND {ans}\n\n")
        return ans - bias

if __name__ == "__main__":
    cam = Camera(init_threshold=True)
    try:
        while True:
            st = time.monotonic()
            cam.update()
            et = time.monotonic()
            t = float((et - st) * 1000)
            fps = round(float(1000 / t))


            print(f"FPS= {fps} || face?: {cam.bool_face_detected} - num_faces: {cam.num_faces_detected} - motion?: {cam.detected_movement} ({cam.movement_avg})")
            # try:
            #     print(cam.movement_zone)
            # except:
            #     pass
    except KeyboardInterrupt:
        cam.exit()