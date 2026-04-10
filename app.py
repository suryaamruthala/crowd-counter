from flask import Flask, render_template, jsonify, Response, request
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import threading #used for running detection in a separate thread so it does not block the main thread
import time
import os
from twilio.rest import Client # Twilio library for sending SMS

app = Flask(__name__)
model = YOLO('yolov8s.pt')
tracker = Tracker()
coco_path = os.path.join(os.path.dirname(__file__), "coco.txt")# used to get class names from the  coco dataset
class_list = open(coco_path).read().split("\n")

people_count = 0
detecting = False
latest_frame = None
detection_thread = None

# Twilio credentials

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SMS_NUMBER = os.getenv("TWILIO_SMS_NUMBER")
TARGET_SMS_NUMBER = os.getenv("TARGET_SMS_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def detect_people():
    global people_count, detecting, latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while detecting:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue
        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if c == "person":
                list.append([x1, y1, x2, y2])
        bbox_idx = tracker.update(list)
        people_count = len(bbox_idx)
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"People Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        latest_frame = frame.copy()
        if people_count == 1:  # Send alert if exactly 1 person detected
            send_sms_alert(people_count)
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

def gen_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count')
def count():
    return jsonify({'count': people_count})

@app.route('/start', methods=['POST'])
def start():
    global detecting, detection_thread
    if not detecting:
        detecting = True
        detection_thread = threading.Thread(target=detect_people, daemon=True)
        detection_thread.start()
        return jsonify({'status': 'started'})
    else:
        return jsonify({'status': 'already running'})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global detecting
    detecting = False
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return 'Server shutting down...'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_sms_alert(count):
    message = f"Alert! Number of people detected: {count}."
    client.messages.create(
        body=message,
        from_=TWILIO_SMS_NUMBER,
        to=TARGET_SMS_NUMBER
    )

if __name__ == '__main__':
    app.run(debug=True)