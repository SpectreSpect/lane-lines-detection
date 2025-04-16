# test2.py

# ====== Monkey Patch for Torch and TorchVision Issues ======
import torch
import types

# Prevent Streamlit's file watcher from iterating torch.__path__
if hasattr(torch, "__path__"):
    torch.__path__ = []

# Ensure torch.onnx exists.
try:
    import torch.onnx
except ModuleNotFoundError:
    torch.onnx = types.ModuleType("torch.onnx")

# Provide a dummy symbolic_opset11 attribute in torch.onnx if it doesn't exist.
if not hasattr(torch.onnx, "symbolic_opset11"):
    torch.onnx.symbolic_opset11 = types.ModuleType("symbolic_opset11")
# =============================================================

import streamlit as st
import tempfile

from src.dataset_balancing import save_plotted_video, get_label_names_dict_inversed
from src.LaneLineModel import LaneLineModel

st.title("Lane Line Detection Video Processor")

# Cache and load model once.
@st.cache_resource
def load_model():
    model_path = "models/sizefull-ep20/model.pt"
    return LaneLineModel(model_path)

lane_model = load_model()

# Cache and load config once.
@st.cache_data
def load_label_dicts():
    return get_label_names_dict_inversed("config.yaml")

label_names_dict_inversed = load_label_dicts()

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Write the uploaded file to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

    # Create an output temporary file.
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = temp_output.name
    temp_output.close()

    st.info("Processing video, please wait...")

    # Process the video.
    save_plotted_video(
        lane_model,
        input_video_path,
        output_video_path,
        label_names_dict_inversed
    )

    st.success("Done! Here's the processed video:")

    # Display video.
    with open(output_video_path, 'rb') as video_file:
        st.video(video_file.read())

    # Download button.
    with open(output_video_path, 'rb') as f:
        st.download_button("Download processed video", f, file_name="processed_video.mp4")
