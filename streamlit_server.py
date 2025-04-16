import streamlit as st
import requests
import tempfile
import os
import uuid

# Set up Streamlit
st.title("Lane Line Detection - Streamlit Interface")

# Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Send video to FastAPI server
    # api_url = "http://localhost:8000/predict/"
    # files = {"file": uploaded_file.getvalue()}
    
    # # Make a request to FastAPI server
    # response = requests.post(api_url, files=files)


    # file_path = response.json()["file_path"]

    # print(file_path)

    st.video("aaaaaa.mp4")

    # Provide download option
    # st.download_button("Download processed video", response.content, file_name="processed_video.mp4")


    # if response.status_code == 200:
    #     st.success("Processing done!")

    #     # Save the video content to a temporary file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
    #         temp_file.write(response.content)
    #         temp_file_path = temp_file.name
        
    #     # uploads_dir = "uploads"
    #     # os.makedirs(uploads_dir)
       
    #     # file_name = str(uuid.uuid4()) + ".mp4"

    #     # file_path = os.path.join(uploads_dir, file_name)

    #     # with open(file_path, "wb") as f:
    #     #     f.write(response.content)


    #     # temp_file.write(response.content)
    #     # temp_file_path = temp_file.name
        
    #     # Display the processed video in Streamlit
    #     # st.video(file_path)

    #     # Provide download option
    #     st.download_button("Download processed video", response.content, file_name="processed_video.mp4")
    # else:
    #     st.error("Error processing the video.")
