import queue
import random
import time

import av
import numpy as np
import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
st.set_page_config(layout="wide")


def main():
    st.title("Rock Paper Scissors")

    if "keras_model" in st.session_state:
        model = st.session_state["keras_model"]
    else:
        # Load the model
        model = load_model(
            "./models/converted_keras_example/keras_Model.h5", compile=False
        )
        st.session_state["keras_model"] = model

    if "class_names" in st.session_state:
        class_names = st.session_state["class_names"]
    else:
        # Load the class labels
        class_names = open(
            "./models/converted_keras_example/labels.txt", "r"
        ).readlines()
        st.session_state["class_names"] = class_names

    if "result" in st.session_state:
        result = st.session_state["result"]
    else:
        result = "set to nothing"
        st.session_state["result"] = result

    def predict():
        image = st.session_state.webcam_image

        if isinstance(image, UploadedFile):
            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1
            data: np.ndarray = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open(image).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            st.session_state["result"] = f"{class_name[2:]}: {confidence_score}"
        else:
            st.session_state["result"] = "skipped"

    # def predict(frame: av.VideoFrame) -> av.VideoFrame:
    # # Create the array of the right shape to feed into the keras model
    # # The 'length' or number of images you can put into the array is
    # # determined by the first position in the shape tuple, in this case 1
    # data: np.ndarray = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # # Get frame as numpy array then convert to PIL.Image for processing
    # frame_numpy = frame.to_ndarray(format="bgr24")
    # image = Image.fromarray(frame_numpy).convert("RGB")

    # # resizing the image to be at least 224x224 and then cropping from the center
    # size = (224, 224)
    # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # # turn the image into a numpy array
    # image_array = np.asarray(image)

    # # Normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # # Load the image into the array
    # data[0] = normalized_image_array

    # # Predicts the model
    # prediction = model.predict(data)
    # index = np.argmax(prediction)
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # # Add prediction and confidence score to result queue
    # result_queue.put(class_name[2:], confidence_score)
    # # print("Class:", class_name[2:], end="")
    # # print("Confidence Score:", confidence_score)
    # return av.VideoFrame.from_ndarray(np.fliplr(frame_numpy), format="bgr24")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.camera_input(
            "webcam_capture",
            key="webcam_image",
            on_change=predict,
            label_visibility="hidden",
        )
        labels_placeholder = st.header("")

    with col2:
        if isinstance(st.session_state.webcam_image, UploadedFile):
            image_paths = {
                "rock": "images/rock.png",
                "paper": "images/paper.png",
                "scissors": "images/scissors.png",
            }

            progress_state = 1.0
            my_bar = st.progress(progress_state, text="ROCK...")
            for text in ["PAPER...", "SCISSORS..."]:
                time.sleep(0.75)
                progress_state -= 1 / 3
                my_bar.progress(progress_state, text=text)

            time.sleep(0.75)
            my_bar.progress(0, text="SHOOT!")

            # randomly pick from rock, paper, scissors
            image_path = random.choice(list(image_paths))
            st.image(image_paths[image_path], width=350)

            labels_placeholder.write(st.session_state.result)
        else:
            st.image("images/rock-paper-scissors.png", width=350)

            # st.write("Predicting...")
            # time.sleep(1)
            # st.write("Predicted: Rock")
        # webrtc_ctx = webrtc_streamer(
        #     key="rps-game",
        #     video_frame_callback=predict,
        #     mode=WebRtcMode.SENDRECV,
        #     media_stream_constraints={"video": True, "audio": False},
        #     async_processing=True,
        # )

        # if webrtc_ctx.state.playing:
        #     labels_placeholder = st.empty()
        #     while True:
        #         try:
        #             result = result_queue.get(timeout=1.0)
        #         except queue.Empty:
        #             result = None
        #         labels_placeholder.write(result)

    # with col2:
    #     image_paths = {
    #         "rock": "images/rock.png",
    #         "paper": "images/paper.png",
    #         "scissors": "images/scissors.png",
    #     }

    #     # button1, button2 = st.columns(2)
    #     # with button1:
    #     play_button = st.button("PLAY", key="play_button", use_container_width=True)
    #     # with button2:
    #     #     clear_button = st.empty()
    #     if play_button:
    #         progress_state = 1.0
    #         my_bar = st.progress(progress_state, text="ROCK...")
    #         for text in ["PAPER...", "SCISSORS..."]:
    #             time.sleep(1)
    #             progress_state -= 1 / 3
    #             my_bar.progress(progress_state, text=text)

    #         time.sleep(1)
    #         my_bar.progress(0, text="SHOOT!")

    #         # randomly pick from rock, paper, scissors
    #         image_path = random.choice(list(image_paths))
    #         st.image(image_paths[image_path], width=350)
    #     else:
    #         st.image("images/rock-paper-scissors.png", width=350)

    # clear_button.button("CLEAR")


if __name__ == "__main__":
    main()
