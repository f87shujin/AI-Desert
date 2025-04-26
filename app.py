import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO

st.title("Vegetable Detection with YOLOv8")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Load a pretrained model

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Run YOLOv8 inference
    results = model(image)
    
    # Get detected objects
    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_objects.append(f"{class_name} ({confidence:.2f})")
    
    # Display detection results
    st.subheader("Detected Objects:")
    if detected_objects:
        for obj in detected_objects:
            st.write(obj)
    else:
        st.write("No objects detected")
