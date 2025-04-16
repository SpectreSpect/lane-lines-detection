from fastapi import FastAPI, File, UploadFile
from src.dataset_balancing import preview_prediction_video, save_plotted_video, get_label_names_dict, get_label_names_dict_inversed
from pydantic import BaseModel
import torch
from src.LaneLineModel import LaneLineModel  # Assuming this is your custom class
import tempfile
import shutil
import cv2
import uuid
import os

# Initialize FastAPI app
app = FastAPI()

# Load your model
model_path = "models/sizefull-ep20/model.pt"
lane_model = LaneLineModel(model_path)

label_names_dict = get_label_names_dict("config.yaml")
label_names_dict_inversed = get_label_names_dict_inversed("config.yaml")

# Define the API request/response format
class VideoResponse(BaseModel):
    output_video_url: str

@app.post("/predict/")
async def predict_video(file: UploadFile = File(...)):

    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    input_video_name = str(uuid.uuid4()) + ".mp4"
    input_video_path = os.path.join(uploads_dir, input_video_name)

    output_video_name = str(uuid.uuid4()) + ".mp4"
    output_video_path = os.path.join(uploads_dir, output_video_name)

    file_bytes = await file.read()
    with open(input_video_path, "wb") as f:
        f.write(file_bytes)

    save_plotted_video(lane_model, input_video_path, output_video_path, label_names_dict_inversed)


    # file.write

    # # Save the uploaded file to a temporary location
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
    #     temp_input.write(await file.read())
    #     input_video_path = temp_input.name

    # # Output file path
    # temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # output_video_path = temp_output.name
    # temp_output.close()




    # # Call your model's method to process the video (example: save_plotted_video)
    # label_names_dict_inversed = {}  # Replace with actual data loading code
    # save_plotted_video(lane_model, input_video_path, output_video_path, label_names_dict_inversed)

    return {"file_path": output_video_path}
    # return "Hello world!"

# Run the FastAPI server (this is not needed in the code, but you can run it from terminal)
# uvicorn fastapi_server:app --reload

