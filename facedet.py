import cv2
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import time
import logging

# Configure logging
logging.basicConfig(filename='activity_detection.log', level=logging.INFO)

# Function to send a message with an image to Telegram
def send_telegram_message_with_photo(bot_token, chat_id, message, photo_path):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(photo_path, 'rb') as photo:
            data = {"chat_id": chat_id, "caption": message}
            response = requests.post(url, files={'photo': photo}, data=data)
            response.raise_for_status()
            logging.info("Message with photo sent successfully.")
    except Exception as e:
        logging.error(f"Error sending message with photo: {e}")

# Function to send a message without an image to Telegram
def send_telegram_message(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=data)
        response.raise_for_status()
        logging.info("Message sent successfully.")
    except Exception as e:
        logging.error(f"Error sending message: {e}")

# Function to save face image
def save_face_image(frame, x, y, w, h, image_count):
    face_img = frame[y:y+h, x:x+w]
    face_path = f"face_{image_count}.jpg"
    cv2.imwrite(face_path, face_img)
    return face_path

# Create a placeholder image for movement detection
def create_movement_placeholder():
    width, height = 200, 200
    background_color = (255, 0, 0)  # Red background
    text_color = (255, 255, 255)    # White text
    
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    text = "Movement Detected"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2
    
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    image.save('movement.jpg')

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up camera
cap = cv2.VideoCapture(0)

# Replace with your Telegram Bot token and chat_id
bot_token = '7203672264:AAED9bhK2C9-ibys0A2s8Dv51YLD5HrP5gU'
chat_id = '6744416532'

# Flags to manage detection state
previous_frame = None
image_count = 0

# Tracking last message times for faces
face_last_seen_time = {}
message_cooldown = 180  # 3 minutes in seconds

# Create placeholder image if it does not exist
if not os.path.exists('movement.jpg'):
    create_movement_placeholder()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    current_time = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        image_count += 1
        face_path = save_face_image(frame, x, y, w, h, image_count)

        # Create a simple unique identifier for the face
        face_id = f"{x}_{y}_{w}_{h}"

        # Check if this face was seen before and the cooldown period has expired
        if face_id not in face_last_seen_time or (current_time - face_last_seen_time[face_id]) > message_cooldown:
            # Determine message based on face detection
            if len(faces) == 1:
                message = "Single face detected in CCTV area!"
            else:
                message = f"Multiple faces detected in CCTV area! Count: {len(faces)}"
            
            send_telegram_message_with_photo(bot_token, chat_id, message, face_path)
            face_last_seen_time[face_id] = current_time  # Update the last seen time

    # Check for movement detection
    if previous_frame is not None:
        diff = cv2.absdiff(previous_frame, gray)
        non_zero_count = cv2.countNonZero(diff)
        current_movement_detected = non_zero_count > 5000  # Adjust threshold as necessary

        if current_movement_detected:
            send_telegram_message(bot_token, chat_id, "Movement detected in CCTV area!")

    # Display the output
    cv2.imshow('Face Detection', frame)
    previous_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
