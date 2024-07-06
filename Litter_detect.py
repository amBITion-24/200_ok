import os
import cv2
import face_recognition
from ultralytics import YOLO
import math
import mediapipe as mp
import numpy as np

# Load the face recognition model and known faces
images_folder_path = '/home/sevengods/Documents/WMS&LD/Litter_detection/Faces/'  # Update with the path to your images folder
known_face_encodings = []
known_face_names = []
for filename in os.listdir(images_folder_path):
    image_file_path = os.path.join(images_folder_path, filename)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        known_image = face_recognition.load_image_file(image_file_path)
        face_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Load the YOLOv5 model
model = YOLO("/home/sevengods/Downloads/runs/detect/train/weights/best.pt")
classnames = ["person", "trash", "litter", "person-littered"]

# Open the video file
video_path = '/home/sevengods/Downloads/24.mp4'  # Update with the path to your video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 1, (frame_width, frame_height))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to draw hand landmarks and connections (updated for MediaPipe 0.9.x)
def draw_landmarks(image, hand_landmarks,trash_box_center):
    # Iterate over detected hands
    for hand_landmarks in hand_landmarks:
        # Check if landmarks exist for this hand
        if hand_landmarks.landmark:
            # Get hand connections (updated for MediaPipe 0.9.x)
            mp_hands.HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmark_index = 8  # Adjust this based on your desired landmark

            # Extract landmark coordinates
            landmark_x = int(hand_landmarks.landmark[landmark_index].x * image.shape[1])
            landmark_y = int(hand_landmarks.landmark[landmark_index].y * image.shape[0])

            # Calculate Euclidean distance to trash box center (assuming you have its coordinates)
            if trash_box_center is not None:
                euclidean_distance = np.sqrt(((landmark_x - trash_box_center[0]) ** 2) + ((landmark_y - trash_box_center[1]) ** 2))

                # Optionally, draw distance text on the image
                cv2.putText(image, f"Distance: {euclidean_distance:.2f} pixels",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Draw red line between landmark and trash box center
                cv2.line(image, (landmark_x, landmark_y), trash_box_center, (0, 0, 255), 2)


while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLOv8 model
    results = model(img, stream=True)
    trash_box_center = None

    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            class_name = classnames[cls]

            # Define color based on class name
            if class_name == "person":
                color = (0, 128, 0)  # Green for person
            elif class_name == "trash":
                color = (25, 0,0 )  # Blue for trash
            elif class_name == "litter":
                color = (0,0,128)
            else:
                color = (255, 255, 255)  # White for unknown class

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3,cv2.FILLED)
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Display class name and confidence
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            if class_name == "trash" or class_name=="litter":
                trash_center_x = (x1 + x2) // 2
                trash_center_y = (y1 + y2) // 2
                if(trash_center_x is not None and trash_center_y is not None):
                    trash_box_center = (trash_center_x, trash_center_y)

            # Convert the BGR image to RGB before processing with MediaPipe
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe hands model
            results = mp_hands.process(rgb_image)

            # Draw detected hand landmarks on the image
            if results.multi_hand_landmarks:
                draw_landmarks(img, results.multi_hand_landmarks,trash_box_center)

            # Perform face recognition if person-littered class is detected
            if class_name == "litter":
                # Find face locations in the frame
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)

                # Iterate over each detected face
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    # Compare the face encoding to known encodings
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # If a match is found, identify the person
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    # Draw a rectangle around the face and display the name
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    if name != "Unknown":
                        # Display the matched face image
                        matched_face_path = os.path.join(images_folder_path, f"{name}.jpg")  # Assuming .jpg format
                        
                        if os.path.exists(matched_face_path):
                            matched_face_img = cv2.imread(matched_face_path)

                            if matched_face_img is not None:        
                                # Resize the matched face image to fit the screen
                                matched_face_img_resized = cv2.resize(matched_face_img, (int(frame_width / 8), int(frame_height/4)))
                                cv2.imshow("Matched Face", matched_face_img_resized)

    # Resize the image to fit the screen
    cv2.namedWindow("Waste Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Waste Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Waste Detection", img)

    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyWindow("Matched Face")
cv2.destroyAllWindows()
