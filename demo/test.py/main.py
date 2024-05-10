import streamlit as st
import pygame
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# Initialize Pygame
pygame.init()
screen_size = (500, 500)
screen = pygame.display.set_mode(screen_size)

# Define the colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)  # Color when a person is detected

# Define crosswalks using pygame Rect
crosswalk1 = pygame.Rect(150, 200, 200, 40)
crosswalk2 = pygame.Rect(150, 300, 200, 40)

# Initialize YOLO model
model = YOLO('yolo-Weights/yolov8m.pt')

def draw_scene(person_detected1, person_detected2):
    screen.fill((0, 0, 0))  # Black background
    # Draw roads
    pygame.draw.rect(screen, GREEN, (100, 190, 300, 60))
    pygame.draw.rect(screen, GREEN, (100, 290, 300, 60))
    # Draw crosswalks
    pygame.draw.rect(screen, YELLOW if person_detected1 else RED, crosswalk1)
    pygame.draw.rect(screen, YELLOW if person_detected2 else RED, crosswalk2)
    # Convert screen to image for Streamlit
    pygame.display.flip()
    view = pygame.surfarray.array3d(screen)
    view = view.transpose([1, 0, 2])
    pil_img = Image.fromarray(view)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return buffer.getvalue()

def app():
    st.title("Real-time Traffic Monitoring")
    col1, col2 = st.columns(2)

    with col1:
        st.header("YOLO Detection")
        vid_frame = st.empty()

    with col2:
        st.header("Intersection Simulation")
        sim_frame = st.empty()

    # Video capture setup
    cap = cv2.VideoCapture('videos/view2.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, size=640, classes=[0])  # Class '0' is typically 'person'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        person_detected1 = False
        person_detected2 = False

        # Analyze detections
        for det in results.xyxy[0]:
            x_center = (det[0] + det[2]) / 2
            y_center = (det[1] + det[3]) / 2
            if crosswalk1.collidepoint(x_center, y_center):
                person_detected1 = True
            if crosswalk2.collidepoint(x_center, y_center):
                person_detected2 = True

        # Update Streamlit frames
        vid_frame.image(frame)
        sim_image = draw_scene(person_detected1, person_detected2)
        sim_frame.image(sim_image, use_column_width=True)

        if st.button("Stop"):
            break

    cap.release()

if __name__ == '__main__':
    app()
