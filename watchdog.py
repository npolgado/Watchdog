#!usr/env/python3
''' 
Watchdog: csv logging of your computers vitals

Prompt:
python script that tracks certain metrics on the computer. the script then takes these metrics and logs them, creating a log file to log events. 

METRICS TO TRACK:
- CPU usage
- GPU usage
- Disk Usage
- Application open
- Microphone Volume and FFT
- Webcam number of faces and detection of movement '''

import datetime
import getpass
import math
import msvcrt
import os
import re
import struct
import subprocess
import sys
import time
from threading import Thread

import numpy as np
import psutil
import pyaudio
import pygame
import webcam

pygame.init()

# constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
LOG_RATE = 10 # time before logging 
ROUNDING = 2

class Observer(object):
    def __init__(self, cpu: float=0.0, gpu: float=0.0, disk: float=0.0, apps: list=[], init_threshold=False) -> None:
        # updated variables
        self.last_cpu = cpu
        self.last_gpu = gpu
        self.last_disk = disk
        self.last_apps = apps

        self.today = datetime.datetime.today().date()
        self.initialized = False

        # open audio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK
        )

        # webcam abstraction
        self.cap = webcam.Camera(init_threshold=init_threshold)

        self.last_cpu, tmp = self.get_cpu_usage()
        self.last_gpu, tmp = self.get_gpu_usage()
        self.last_disk, tmp = self.get_disk_usage()
        self.last_apps = self.get_apps_open()
        self.get_volume_fft()
        self.get_webcam_movement()

        cwd = os.getcwd()
        logs_path = os.path.join(cwd, 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        self.log_path = ""
        
        self.initialized = True
        # self.key_th = Thread(target=lambda: self.get_keyboard_events())
        # self.key_th.start()

    def exit(self):
        self.cap.exit()
        self.p.close(self.stream)

    def num_extract(self, string):
        ''' given a string, return a an array of all numbers in the string. 
        
        NOTE: can throw errors, in which can will return the string back
        '''
        try:
            return re.findall(r'\d+', string)
        except Exception as e:
            print(f"ERROR num_extract\n{e}")
            return string

    def find_app_difference(self, curr: list, prev: list):
        ''' given too lists, return a list of uncommon values'''
        diff = []
        for i in curr:
            if i not in prev: diff.append(i)
        for i in prev:
            if i not in curr and i not in diff: diff.append(i)
        return diff    

    def get_cpu_usage(self):
        ''' function to get the current CPU usage '''
        # get the current cpu usage
        # global last_cpu
        # print(f"{type(last_cpu)}\n{last_cpu}")
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            if self.initialized: cpu_change = round(float(cpu_usage - self.last_cpu),ROUNDING)
            else: cpu_change = 0.0
            self.last_cpu = cpu_usage
        except Exception as e:
            print(f"\nERROR getting cpu usage\n{e}")
            pass
        return cpu_usage, cpu_change

    def get_gpu_usage(self):
        ''' function to get the current GPU usage '''

        # get the current gpu usage
        # global last_gpu
        gpu_usage = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv").read()
        try:
            # gpu_usage = gpu_usage.split('\n')[1]
            # gpu_usage = gpu_usage.split(',')[1]
            gpu_usage = round(float(self.num_extract(gpu_usage)[0]),ROUNDING)
            if self.initialized: gpu_change = round(float(gpu_usage - self.last_gpu),ROUNDING)
            else: gpu_change = 0.0
            self.last_gpu = gpu_usage
        except Exception as e:
            gpu_change = 0
            self.last_gpu = gpu_usage
            print(f"\nERROR string manipulations\n{e}")
            pass
        return gpu_usage, gpu_change

    def get_disk_usage(self):
        ''' function to get the current disk usage '''
        # get the current disk usage
        # global last_disk
        try:
            disk_usage = round(float(psutil.disk_usage('/').percent),ROUNDING)
            if self.initialized: disk_change = float(disk_usage - self.last_disk)
            else: disk_change = 0.0
            self.last_disk = disk_usage
        except Exception as e:
            disk_usage = 0
            self.last_disk = disk_usage
            print(f"\nERROR grabbing disk usage\n{e}")
            pass
        return disk_usage, disk_change

    def get_apps_open(self):
        ''' function to get the applications that are currently open '''
        '''# get a list of open applications
        try:
            apps_open = []
            # for proc in psutil.process_iter():
            #     if proc.name() != 'System Idle Process':
            #         apps_open.append(proc.name())
            user_apps=subprocess.run(['ps', '-U', '$USER', '-o', 'comm'], stdout=subprocess.PIPE, universal_newlines=True)
            for app in user_apps.stdout.splitlines():
                apps_open.append(app)
        except Exception as e:
            print(f"\nERROR grabbing applications open\n{e}")
            pass
        return apps_open'''
        try:
            running_apps = []
            for proc in psutil.process_iter():
                try:
                    if proc.name() != 'System Idle Process':
                        running_apps.append(proc.name())
                    # pinfo = proc.as_dict(attrs=['pid', 'name'])
                    # if os.name == 'nt':
                    #     # Windows
                    #     running_apps.append(pinfo['name'])
                    # elif os.name == 'posix':
                    #     # Linux
                    #     running_apps.append(pinfo['pid'])
                except Exception:
                    pass
            change = self.find_app_difference(running_apps, self.last_apps)
            self.last_apps = running_apps
        except Exception as e:
            change = []
            print(f"ERROR getting apps\n{e}")
        return change

    def get_volume_fft(self):
        ''' function to get the current microphone volume and FFT '''

        # get the current microphone volume
        try:
            # get audio data
            data = np.frombuffer(self.stream.read(CHUNK), dtype=np.int16)

            # RMS
            rms = round(float(np.sqrt(np.mean(data**2))),ROUNDING)
            
            # calculate the volume in dB
            dB = round(float(20 * np.log10(rms)),ROUNDING)
            
            # calculate the fft
            fft = abs(np.fft.rfft(data))
            max_frequency = round(float(fft[np.argmax(fft)]),ROUNDING)
            
            return (dB, rms, max_frequency)

        except Exception as e:
            print(f"\nERROR analyzing microphone\n{e}")
            pass
        # return the volume and fft
        # return volume_db, fourier
        return (dB, rms, max_frequency)

    def get_webcam_movement(self):
        ''' function to get the current webcam movement and number of faces
        
        currently movement is replaced with bool(faces were detected)
        '''
        # get the current webcam movement
        self.cap.update()

        motion = int(self.cap.detected_movement)
        num_faces = int(self.cap.num_faces_detected)
        # _, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.GaussianBlur(frame, (21, 21), 0)

        # # get the number of faces detected
        # faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
        # num_faces = len(faces)

        # # get the motion detection
        # st = time.monotonic()
        # et = time.monotonic()
        # dt = float(et - st)
        # timeout = 1 # 2 min
        # first_frame = None
        # motion = 0
        # while dt <= timeout:
        #     _, frame = cap.read()
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     gray = cv2.GaussianBlur(gray, (21, 21), 0)
        #     if first_frame is None:
        #         first_frame = gray
        #         continue
        #     delta_frame = cv2.absdiff(first_frame, gray)
        #     thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        #     thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        #     (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for contour in cnts:
        #         if cv2.contourArea(contour) < 10000:
        #             continue
        #         motion = 1
            
        #     et = time.monotonic()
        #     dt = float(et - st)

        # return the motion and number of faces
        return motion, num_faces

    def get_keyboard_events(self):
        # self.keyboard = msvcrt.getch()
        # print(bool(msvcrt.kbhit()))
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        self.keyboard = True
                self.keyboard =  False
        except Exception as e:
            print(f"ERROR keyboard thread\n{e}")

    def log_event(self, event, is_header=False):
        ''' function to log an event to a log file '''
        timestamp = datetime.datetime.now().time()
        hours = timestamp.hour
        minutes = timestamp.minute
        seconds = timestamp.second

        cwd = os.getcwd()
        self.log_path = os.path.join(cwd, f'logs/{self.today}.csv')
        ignore_header = False
        if os.path.isfile(self.log_path): ignore_header = True
        # write the log to the log file
        log_file = open(self.log_path, 'a+')
        if is_header:
            if ignore_header: pass
            else: log_file.write(event)
        else: 
            log_file.write(f'{hours},{minutes},{seconds},{event}')
        log_file.close()

if __name__ == "__main__":
    eye = Observer(init_threshold=False)
    eye.log_event('HOUR,MINUTE,SECOND,CPU_PERCENT,CHANGE_IN_CPU,GPU_PERCENT,CHANGE_IN_GPU,DISK_PERCENT,CHANGE_IN_DISK,VOLUME,RMS,FREQ,MOTION,NUM_FACES\n', is_header=True)
    time.sleep(1)
    while True:
        try:
            # get the current metrics
            cpu_usage, dcpu = eye.get_cpu_usage()
            gpu_usage, dgpu = eye.get_gpu_usage()
            disk_usage, ddisk = eye.get_disk_usage()
            # new_apps = eye.get_apps_open()
            # keyboard = eye.get_keyboard_events()
            volume, rms, max_freq = eye.get_volume_fft()
            motion, num_faces = eye.get_webcam_movement()

            # log the metrics
            eye.log_event(f'{cpu_usage},{dcpu},{gpu_usage},{dgpu},{disk_usage},{ddisk},{volume},{rms},{max_freq},{motion},{num_faces}\n')
            time.sleep(LOG_RATE)
        except KeyboardInterrupt as e:
            try:
                eye.exit()
                sys.exit()
            except Exception as e:
                print(f"ERROR exiting\n{e}")