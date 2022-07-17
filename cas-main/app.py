from flask import Flask, render_template, Response
from sys import stdout
import logging
import cv2
from makeup_artist import Makeup_artist
import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import  datetime
from camera import Camera
import base64
from utils import base64_to_pil_image, pil_image_to_base64
from PIL import Image
from io import StringIO
from io import BytesIO
from flask_socketio import SocketIO
path = 'employee images'
employeeImg = []
employeeName = []
myList = os.listdir(path)


def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

def readb64(base64_string):
    #sbuf = StringIO(base64_string)
    #sbuf.write(base64.b64decode(base64_string))
    #sbuf = sbuf.read().encode('utf8')
    #sbuf.write(base64_string)
    sbuf = BytesIO(base64_string)
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            

for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    employeeImg.append(curimg)
    employeeName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(employeeImg)

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())  # use 0 for web camera
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        frame = camera.get_frame()
        frame = readb64(frame)
        
        
        
        
        

        facesInFrame = face_rec.face_locations(frame)
        encodeFacesInFrame = face_rec.face_encodings(frame, facesInFrame)
        
        print(frame)

        for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
            
            matches = face_rec.compare_faces(EncodeList, encodeFace)
            facedis = face_rec.face_distance(EncodeList, encodeFace)
            print('facedis',facedis)
            
            matchIndex = np.argmin(facedis)
            if np.argmin(facedis) < 1:
                
                name = employeeName[matchIndex].upper()
                #y1, x2, y2, x1 = faceloc
                #y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                #cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
                #cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                startX, startY, endX, endY = faceloc
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                MarkAttendence(name)
                print(name)
                

        
        
        
        
        
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    #app.run(debug=False)
    socketio.run(app)
