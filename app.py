import cv2
import math
import argparse
import tkinter as tk
from tkinter import ttk
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import urllib.request
from werkzeug.utils import secure_filename

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return render_template("index.html", uploaded_image=image.filename)
    return render_template("index.html")


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')
    args=parser.parse_args()
    faceProto="D:/gad/opencv_face_detector.pbtxt"
    faceModel="D:/gad/opencv_face_detector_uint8.pb"
    ageProto="D:/gad/age_deploy.prototxt"
    ageModel="D:/gad/age_net.caffemodel"
    genderProto="D:/gad/gender_deploy.prototxt"
    genderModel="D:/gad/gender_net.caffemodel"
    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-5)', '(5-10)', '(10-15)', '(15-20)', '(20-25)', '(25-35)','(35-45)' '(45-55)', '(55-70)', '(75-100)']
    genderList=['Male','Female']
    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    padding = 20
    frame=cv2.imread(filename)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
         :min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imwrite('test.jpg', resultImg)
    return send_from_directory(app.config["IMAGE_UPLOADS"], 'test.jpg')

if __name__ == "__main__":
    app.run(host='127.0.0.2', port=5004, debug=False)