import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from PIL import Image
from io import BytesIO

import queue
from pathlib import Path
from typing import List, NamedTuple

import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from ultralytics import YOLO

from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer



st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# STUN Server
STUN_STRING = "stun:" + str(getSTUNServer())
STUN_SERVER = [{"urls": [STUN_STRING]}]

# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Realtime")

st.write("Detect the road damage in realtime using USB Webcam. This can be useful for on-site monitoring with personel on the ground. Select the video input device and start the inference.")

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    
    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results on the queue
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    return av.VideoFrame.from_ndarray(_image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

if image_file is not None:

    # Load the image
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)

    # Perform inference
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        downloadButton = st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )


st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _frame_count/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds)

        st.write("Video Duration :", _duration_strings)
        st.write("Width, Height and FPS :", _width, _height, _fps)

        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # Read until video is completed
        _frame_counter = 0
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()
            if ret == True:
                
                # Convert color-chanel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform inference
                _image = np.array(frame)

                image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
                results = net.predict(image_resized, conf=score_threshold)
                
                # Save the results
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    detections = [
                    Detection(
                        class_id=int(_box.cls),
                        label=CLASSES[int(_box.cls)],
                        score=float(_box.conf),
                        box=_box.xyxy[0].astype(int),
                        )
                        for _box in boxes
                    ]

                annotated_frame = results[0].plot()
                _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation = cv2.INTER_AREA)

                print(_image_pred.shape)
                
                # Write the image to file
                _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
                cv2writer.write(_out_frame)
                
                # Display the image
                imageLocation.image(_image_pred)

                _frame_counter = _frame_counter + 1
                inferenceBar.progress(_frame_counter/_frame_count, text=inferenceBarText)
            
            # Break the loop
            else:
                inferenceBar.empty()
                break

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    # Download button for the video
    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        # Also rerun the appplication after download
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
    with col2:
        if st.button('Restart Apps', use_container_width=True, type="primary"):
            # Rerun the application
            st.rerun()

st.title("Road Damage Detection - Video")
st.write("Detect the road damage in using Video input. Upload the video and start detecting. This section can be useful for examining and process the recorded videos.")

video_file = st.file_uploader("Upload Video", type=".mp4", disabled=st.session_state.runningInference)
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction. You can change the threshold before running the inference.")

if video_file is not None:
    if st.button('Process Video', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = "Processing Video " + video_file.name
        st.warning(_warning)
        processVideo(video_file, score_threshold)
