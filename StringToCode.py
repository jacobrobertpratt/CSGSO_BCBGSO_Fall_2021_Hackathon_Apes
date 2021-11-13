import PySimpleGUI as sg
from tkinter import *
import subprocess
import os
import time
import cv2

root = Tk()
root.geometry("400x400")

combined = "i = 0\nfor i in range(10):\n\ti+=1\n\tprint('hi')"

def runPython(program):
    tempFile = open("MainMethod.py", 'w')
    tempFile.write(program)
    tempFile.close()
    startPython()

def runJava(program):
    tempFile = open("Method.java", 'w')
    tempFile.write(program)
    tempFile.close()
    compileJava()
    time.sleep(.5)
    startJava()

def startPython():
    cmd = 'python ' + "MainMethod.py"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    suboutput = process.stdout.read()
    print(suboutput.decode("utf-8"))

def compileJava():
    cmd = 'javac ' + "Method.java"
    process = subprocess.Popen(cmd, shell=True)

def startJava():
    cmd = 'java ' + "Method"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    suboutput = process.stdout.read()
    print(suboutput.decode("utf-8"))

def startFeed():
    cam = cv2.VideoCapture(0)  # 0 -> index of camera

    i = 0
    while i<2:
        s, img = cam.read()
        if s:  # frame captured without any errors
            # cv2.namedWindow("cam-test")
            # cv2.imshow("cam-test", img)
            cv2.waitKey(1)
            cv2.imwrite("filename.jpg", img)

            scale_percent = 200
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            # cv2.imshow("Resized image", resized)
            cv2.waitKey(1)
            cv2.imwrite("filename.jpg", resized)


            i+=1

    # Generate string from image here.

def click():
    startFeed()

button = Button(root, text="Run Code From Camera", command=click, width=20, height=10, bg="black", fg="white", borderwidth=5)
button.pack()
root.mainloop()
