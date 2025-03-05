import onnxruntime
import numpy as np
import gradio as gr
from PIL import Image

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Load ONNX model
session = onnxruntime.InferenceSession('dr-model.onnx')

def transform_image(image):
    # Resize
    image = image.resize((224, 224))
    # Convert to numpy array and explicitly set dtype to float32
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Normalize using same parameters as before
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose to channel-first format (NCHW)
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32)  # Ensure final output is float32

def predict(input_img):
    """Predict DR grade from input image using ONNX model"""
    input_tensor = transform_image(input_img)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: input_tensor})[0][0]
    
    # Apply softmax
    exp_preds = np.exp(prediction - np.max(prediction))
    probabilities = exp_preds / exp_preds.sum()
    
    confidences = {labels[i]: float(probabilities[i]) for i in labels}
    return confidences

# Rest of the Gradio interface remains the same
dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Detection",
    description="This app uses a quantized ONNX model for DR detection.",
    examples=[
        "sample/10_left.jpeg",
        "sample/10_right.jpeg",
        "sample/15_left.jpeg",
        "sample/16_right.jpeg",
    ],
    analytics_enabled=False,
)
if __name__ == "__main__":
    dr_app.launch()