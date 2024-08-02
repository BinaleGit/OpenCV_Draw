import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Variables to store drawing state and previous positions for both hands
drawing = [False, False]
erasing = [False, False]
prev_x, prev_y = [None, None], [None, None]
smooth_factor = 5

# Deque to store the previous positions for smoothing
points = [deque(maxlen=smooth_factor), deque(maxlen=smooth_factor)]

# List of colors and their names
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_names = ["Green", "Blue", "Red", "Yellow", "Cyan", "Magenta"]
color_index = 0
current_color = colors[color_index]
current_color_name = color_names[color_index]

# Line thickness
line_thickness = 5
eraser_thickness = 50

# List to store squares
squares = []

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
def smooth_coordinates(x, y, points):
    points.append((x, y))
    avg_x = int(np.mean([pt[0] for pt in points]))
    avg_y = int(np.mean([pt[1] for pt in points]))
    return avg_x, avg_y

# Mouse callback function for square drawing
drawing_square = False
pt1 = (0, 0)
pt2 = (0, 0)

def draw_square(event, x, y, flags, param):
    global pt1, pt2, drawing_square

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing_square:
            pt1 = (x, y)
            drawing_square = True
        else:
            squares.append((pt1, (x, y)))
            drawing_square = False

# Open the webcam
cap = cv2.VideoCapture(0)

# Function to display instructions in a separate window
def show_instructions():
    instruction_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    instructions = [
        "Instructions:",
        "c - Change Color",
        "e - Toggle Eraser",
        "+ - Increase Thickness",
        "- - Decrease Thickness",
        "q - Quit",
        "s - Draw Square",
        "backspace - Clear Canvas"
    ]
    y0, dy = 50, 30
    for i, line in enumerate(instructions):
        y = y0 + i*dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow('Instructions', instruction_image)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize the frame for a larger display
    frame = cv2.resize(frame, (1280, 720))

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect the hands
    results = hands.process(frame_rgb)

    # Convert the image color back so it can be displayed
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Initialize canvas if not already
    if 'canvas' not in locals():
        canvas = 255 * np.ones_like(frame)  # Create a white canvas

    # Draw the hand annotations on the image and count fingers for each hand
    left_fingers = 0
    right_fingers = 0
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            finger_count = count_fingers(hand_landmarks)

            # Check if the hand is left or right
            label = handedness.classification[0].label
            if label == 'Left':
                left_fingers = finger_count
                hand_index = 0
            else:
                right_fingers = finger_count
                hand_index = 1

            # Drawing logic: if the specific gesture (e.g., 1 finger up) is detected, start drawing
            if finger_count == 1:  # Assuming the gesture has 1 finger up
                drawing[hand_index] = True
                # Get the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                x, y = smooth_coordinates(x, y, points[hand_index])
                if prev_x[hand_index] is not None and prev_y[hand_index] is not None:
                    # Draw a line from the previous point to the current point
                    if erasing[hand_index]:
                        cv2.line(canvas, (prev_x[hand_index], prev_y[hand_index]), (x, y), (255, 255, 255), eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x[hand_index], prev_y[hand_index]), (x, y), current_color, line_thickness)
                prev_x[hand_index], prev_y[hand_index] = x, y
            else:
                drawing[hand_index] = False
                prev_x[hand_index], prev_y[hand_index] = None, None

    # Draw all squares on the canvas
    for sq in squares:
        cv2.rectangle(canvas, sq[0], sq[1], (0, 255, 0), 2)

    # Display the finger count for each hand
    cv2.putText(frame, f'Left Fingers: {left_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right Fingers: {right_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Combine the frame and the canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show square drawing mode and first click indicator
    if drawing_square:
        cv2.putText(combined, "Square Drawing Mode", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, "Square Drawing Mode", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if pt1 != (0, 0):
            cv2.circle(combined, pt1, 5, (0, 0, 255), -1)

    # Display the combined frame
    cv2.imshow('Hand Tracking and Drawing', combined)

    # Display the instructions
    show_instructions()

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
        # Toggle erasing mode for both hands
        erasing = [not erasing[0], not erasing[1]]
    elif key == ord('+') and line_thickness < 20:
        # Increase line thickness
        if any(erasing):
            eraser_thickness += 5
        else:
            line_thickness += 1
    elif key == ord('-') and line_thickness > 1:
        # Decrease line thickness
        if any(erasing) and eraser_thickness > 10:
            eraser_thickness -= 5
        else:
            line_thickness -= 1
    elif key == ord('s'):
        # Toggle square drawing mode
        if not drawing_square:
            cv2.setMouseCallback('Hand Tracking and Drawing', draw_square)
            drawing_square = True
            pt1 = (0, 0)
        else:
            cv2.setMouseCallback('Hand Tracking and Drawing', lambda *args: None)
            squares.append((pt1, pt2))
            drawing_square = False
            pt1 = (0, 0)
            pt2 = (0, 0)
    elif key == 8:  # Backspace key
        # Clear the canvas
        canvas = 255 * np.ones_like(frame)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
