import gradio as gr
import os
from src.dataset_balancing import save_plotted_video, get_label_names_dict_inversed
from src.LaneLineModel import LaneLineModel
import shutil
import uuid

# Load model and config once at startup
model_path = "models/sizefull-ep20/model.pt"
lane_model = LaneLineModel(model_path)
label_names_dict_inversed = get_label_names_dict_inversed("config.yaml")

# Processing function
def process_video(video_file):
    input_path = video_file
    output_path = str(uuid.uuid4()) + ".mp4"

    # Call your function
    save_plotted_video(lane_model, input_path, output_path, label_names_dict_inversed)

    return output_path

# Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Road Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Lane Line Detection",
    description="Upload a road video to process it using the lane line model.",
)

if __name__ == "__main__":
    demo.launch()