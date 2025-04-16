import dash
from dash import html, dcc, Input, Output, State
import base64
import tempfile
import os

from src.dataset_balancing import save_plotted_video, get_label_names_dict_inversed
from src.LaneLineModel import LaneLineModel

# Pre-load your model and label dictionary once
model_path = "models/sizefull-ep20/model.pt"
lane_model = LaneLineModel(model_path)
label_names_dict_inversed = get_label_names_dict_inversed("config.yaml")

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    html.H1("Lane Line Detection Video Processor"),
    dcc.Upload(
        id='upload-video',
        children=html.Div([
            'Drag and Drop or ', html.A('Select a Video File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        accept="video/mp4,video/quicktime,video/x-msvideo"
    ),
    html.Button('Process Video', id='process-button', n_clicks=0),
    html.Div(id='output-message'),
    html.Div(id='video-output')
])

@app.callback(
    [Output('video-output', 'children'),
     Output('output-message', 'children')],
    [Input('process-button', 'n_clicks')],
    [State('upload-video', 'contents'),
     State('upload-video', 'filename')]
)
def process_video(n_clicks, video_contents, filename):
    if n_clicks > 0 and video_contents is not None:
        # The uploaded file is received as a base64-encoded string.
        content_type, content_string = video_contents.split(',')
        decoded = base64.b64decode(content_string)

        # Write the uploaded video to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
            tmp_in.write(decoded)
            input_video_path = tmp_in.name

        # Set an output temporary file path.
        output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")

        # Process the video using your existing function.
        save_plotted_video(lane_model, input_video_path, output_video_path, label_names_dict_inversed)

        # Read and encode the processed video in base64.
        with open(output_video_path, "rb") as f:
            processed_video = f.read()
        video_encoded = base64.b64encode(processed_video).decode('ascii')
        video_src = "data:video/mp4;base64," + video_encoded

        # Optionally remove temporary files.
        os.remove(input_video_path)
        os.remove(output_video_path)

        return (
            html.Video(
                controls=True,
                src=video_src,
                style={'width': '100%', 'height': 'auto'}
            ),
            "Video processing complete."
        )
    return "", ""

if __name__ == '__main__':
    app.run(debug=True)
