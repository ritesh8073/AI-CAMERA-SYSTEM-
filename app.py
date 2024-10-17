from flask import Flask, render_template, request, redirect, url_for, Response
import threading
import cv2
import numpy as np
import requests
import logging
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')

logging.basicConfig(filename='activity_detection.log', level=logging.INFO)

surveillance_running = False
bot_token = ""
chat_id = ""

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def send_telegram_message(bot_token, chat_id, message, image_path=None):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=data)
        response.raise_for_status()
        logging.info("Message sent successfully.")

        if image_path:
            with open(image_path, 'rb') as f:
                requests.post(f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                              data={"chat_id": chat_id}, files={"photo": f})
                logging.info("Image sent successfully.")
    except Exception as e:
        logging.error(f"Error sending message: {e}")

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = [(boxes[i], class_ids[i]) for i in indexes.flatten()] if len(indexes) > 0 else []
    return detected_objects

def generate_frames():
    global surveillance_running, bot_token, chat_id
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while surveillance_running:
        success, frame = cap.read()
        if not success:
            logging.error("Failed to grab frame")
            continue

        detections = detect_objects(frame)

        for (box, class_id) in detections:
            x, y, w, h = box
            label = str(classes[class_id])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_path = "detected.jpg"
            cv2.imwrite(img_path, frame)
            send_telegram_message(bot_token, chat_id, f"Detected: {label}", img_path)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', surveillance_running=surveillance_running)

@app.route('/start', methods=['POST'])
def start():
    global surveillance_running, bot_token, chat_id

    bot_token = request.form['bot_token']
    chat_id = request.form['chat_id']
    surveillance_running = True
    threading.Thread(target=generate_frames).start()

    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop():
    global surveillance_running
    surveillance_running = False
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
