import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model = YOLO(r'best.pt')

st.title("Medical Image Segmentation")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Button to start the segmentation
    if st.button('Process Image'):
        with st.spinner("Processing..."):
            # Save the image temporarily
            temp_image_path = "temp_image.png"
            cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Predict on the input image
            results = model.predict(temp_image_path)
            boxes = results[0].boxes  
            masks = results[0].masks  
            class_names = model.names  
            predictions = []
            for box, mask in zip(boxes, masks):
                class_id = int(box.cls)  
                confidence = box.conf  
                segmentation = mask.data  
                predictions.append((confidence, class_id, segmentation, box))
            predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
            grouped_predictions = {}
            for pred in predictions:
                conf, class_id, segmentation, box = pred
                if class_id not in grouped_predictions:
                    grouped_predictions[class_id] = []
                grouped_predictions[class_id].append(pred)
            if len(grouped_predictions) == 1:
                selected_predictions = sum(grouped_predictions.values(), [])
            else:
                highest_confidence_predictions = [max(preds, key=lambda x: x[0]) for preds in grouped_predictions.values()]
                selected_predictions = [max(highest_confidence_predictions, key=lambda x: x[0])]
            overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            for idx, (confidence, class_id, segmentation, box) in enumerate(selected_predictions):
                class_name = class_names[class_id] if class_id in class_names else f"Class {class_id}"
                segmentation_mask = segmentation.cpu().numpy() 
                if segmentation_mask.ndim == 3:
                    segmentation_mask = segmentation_mask.squeeze(0)  # Remove extra dimensions if necessary
                segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
                segmentation_mask = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]))
                colored_mask = np.zeros_like(overlay)  
                color = colors[idx % len(colors)]
                colored_mask[:, :, 0] = (segmentation_mask / 255) * color[0]  
                colored_mask[:, :, 1] = (segmentation_mask / 255) * color[1]  
                colored_mask[:, :, 2] = (segmentation_mask / 255) * color[2] 
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)  
                label = f"{class_name}: {confidence.item():.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            output_path = "segmented_output.png"
            cv2.imwrite(output_path, overlay)
            st.image(output_path, caption="Processed Image", use_column_width=True)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Segmented Image",
                    data=file,
                    file_name="segmented_output.png",
                    mime="image/png"
                )

        st.success("Processing complete!")
