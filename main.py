import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Variables to store drawing state and previous positions
drawing = False
erasing = False
prev_x, prev_y = None, None
smooth_factor = 5

# Deque to store the previous positions for smoothing
points = deque(maxlen=smooth_factor)

# List of colors and their names
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_names = ["Green", "Blue", "Red", "Yellow", "Cyan", "Magenta"]
color_index = 0
current_color = colors[color_index]
current_color_name = color_names[color_index]

# Line thickness
line_thickness = 5
eraser_thickness = 50

# Function to count fingers
def count_fingers(hand_landmarks):
    # List of tips of fingers
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    count = 0

    # Count fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    # Check thumb (assuming it's checking for a specific gesture)
    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 2].x:
        count += 1

    return count

# Function to smooth coordinates using a moving average
def smooth_coordinates(x, y):
    points.append((x, y))
    avg_x = int(np.mean([pt[0] for pt in points]))
    avg_y = int(np.mean([pt[1] for pt in points]))
    return avg_x, avg_y

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a blank white canvas
canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize the frame for a larger display
    frame = cv2.resize(frame, (1280, 720))

    # Flip the frame horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the frame and detect the hands
    results = hands.process(frame)

    # Convert the image color back so it can be displayed
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Initialize canvas if not already
    if canvas is None:
        canvas = 255 * np.ones_like(frame)  # Create a white canvas

    # Draw the hand annotations on the image and count fingers for each hand
    left_fingers = 0
    right_fingers = 0
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            finger_count = count_fingers(hand_landmarks)

            # Check if the hand is left or right
            label = handedness.classification[0].label
            if label == 'Left':
                left_fingers = finger_count
            else:
                right_fingers = finger_count

            # Drawing logic: if the specific gesture (e.g., 1 finger up) is detected, start drawing
            if finger_count == 1:  # Assuming the gesture has 1 finger up
                drawing = True
                # Get the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                x, y = smooth_coordinates(x, y)
                if prev_x is not None and prev_y is not None:
                    # Draw a line from the previous point to the current point
                    if erasing:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, line_thickness)
                prev_x, prev_y = x, y
            else:
                drawing = False
                prev_x, prev_y = None, None

    # Display the finger count for each hand
    cv2.putText(frame, f'Left Fingers: {left_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right Fingers: {right_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Combine the frame and the canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display current mode and instructions
    mode_text = "Mode: Erasing" if erasing else "Mode: Drawing"
    color_text = f"Color: {current_color_name}"
    thickness_text = f"Thickness: {eraser_thickness if erasing else line_thickness}"
    instructions = "Keys: 'c' - Change Color, 'e' - Toggle Eraser, '+' - Increase Thickness, '-' - Decrease Thickness, 'q' - Quit"
    cv2.putText(combined, mode_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, mode_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined, color_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, color_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined, thickness_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, thickness_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined, instructions, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, instructions, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the combined frame with instructions
    cv2.imshow('Hand Tracking and Drawing', combined)

    # Handle keyboard input
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Change color
        color_index = (color_index + 1) % len(colors)
        current_color = colors[color_index]
        current_color_name = color_names[color_index]
    elif key == ord('e'):
        # Toggle erasing mode
        erasing = not erasing
    elif key == ord('+') and line_thickness < 20:
        # Increase line thickness
        if erasing:
            eraser_thickness += 5
        else:
            line_thickness += 1
    elif key == ord('-') and line_thickness > 1:
        # Decrease line thickness
        if erasing and eraser_thickness > 10:
            eraser_thickness -= 5
        else:
            line_thickness -= 1

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
