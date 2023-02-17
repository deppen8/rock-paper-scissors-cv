import random
import time
from pathlib import Path

import av
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from streamlit_extras.let_it_rain import rain
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(layout="wide")

base_path = Path(__file__).parents[0]

st.write(sorted(list(Path(__file__).glob("**/*"))))


def main():
    st.radio(
        label="Select an AI model flavor",
        options=["Lots of training data", "Little training data"],
        horizontal=True,
        key="model_flavor",
    )

    # Logic to load the model or grab it from memory if it has already been loaded
    # There are two model flavors, one trained on lots of data and one trained on little data
    if st.session_state["model_flavor"] == "Lots of training data":
        if "keras_model_lots_of_training_data" in st.session_state:
            model = st.session_state["keras_model_lots_of_training_data"]
        else:
            # Load the model
            model = load_model(
                base_path / "models" / "lots_of_training_data" / "keras_Model.h5",
                compile=False,
            )
            st.session_state["keras_model_lots_of_training_data"] = model

        if "class_names" in st.session_state:
            class_names = st.session_state["class_names"]
        else:
            # Load the class labels
            class_names = open(
                base_path / "models" / "lots_of_training_data" / "labels.txt", "r"
            ).readlines()
            st.session_state["class_names"] = class_names

    elif st.session_state["model_flavor"] == "Little training data":
        if "keras_model_little_training_data" in st.session_state:
            model = st.session_state["keras_model_little_training_data"]
        else:
            # Load the model
            model = load_model(
                base_path / "models" / "little_training_data" / "keras_Model.h5",
                compile=False,
            )
            st.session_state["keras_model_little_training_data"] = model

        if "class_names" in st.session_state:
            class_names = st.session_state["class_names"]
        else:
            # Load the class labels
            class_names = open(
                base_path / "models" / "little_training_data" / "labels.txt", "r"
            ).readlines()
            st.session_state["class_names"] = class_names

    def predict(frame: av.VideoFrame) -> tuple[str, float]:
        """Run inference on a single frame.

        Args:
            frame (av.VideoFrame): A single frame from the video feed.

        Returns:
            tuple[str, float]: A class prediction string and confidence score.
        """
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data: np.ndarray = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Get frame as numpy array then convert to PIL.Image for processing
        frame_numpy = frame.to_ndarray(format="bgr24")
        image = Image.fromarray(frame_numpy).convert("RGB")

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

        return class_name[2:].rstrip(), confidence_score

    col1, col2 = st.columns([4, 3], gap="medium")
    with col1:
        # Initialize SENDRECV feed
        webrtc_ctx = webrtc_streamer(
            key="input_feed",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
        )

        # Initialize SENDONLY feed that pulls from the SENDRECV feed
        # This is a hack so that it is easier to grab a single frame.
        webrtc_ctx_sendonly = webrtc_streamer(
            key="output_feed",
            mode=WebRtcMode.SENDONLY,
            source_video_track=webrtc_ctx.output_video_track,
            desired_playing_state=webrtc_ctx.state.playing,
            video_receiver_size=4,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        # Initialize some UI elements
        play_button = st.button(
            "PLAY", key="play_button", type="primary", use_container_width=True
        )
        my_bar = st.progress(1.0, text="READY!")

        countdown_images = st.empty()
        result_summary = st.title("")

        with st.container():
            little_col1, little_col2 = st.columns([1, 1])
            with little_col1:
                human_image_box = st.empty()
            with little_col2:
                result_class_txt = st.header("")
                result_confidence_txt = st.header("")

        with st.container():
            little_col3, little_col4 = st.columns([1, 1])
            with little_col3:
                cpu_image_box = st.empty()
            with little_col4:
                cpu_class_txt = st.header("")

        OBJECTS = ["rock", "paper", "scissors"]

        IMAGE_PATHS = {
            "rock": "images/rock.png",
            "paper": "images/paper.png",
            "scissors": "images/scissors.png",
            "shoot": "images/shoot.png",
        }
        if play_button:  # When the play button is pressed
            # Start countdown and cycle through images
            progress_state = 1.0
            my_bar.progress(progress_state, text="ROCK...")
            countdown_images.image("images/rock.png", use_column_width=True)
            for text in ["PAPER...", "SCISSORS...", "SHOOT..."]:
                time.sleep(1.0)
                progress_state -= 1 / 3
                my_bar.progress(progress_state, text=text)
                countdown_images.image(
                    IMAGE_PATHS[text[:-3].lower()], use_column_width=True
                )

            time.sleep(1.5)
            my_bar.progress(0, text="SHOOT...")
            time.sleep(2.0)

            countdown_images.empty()

            # Get a single frame from the SENDONLY feed
            if webrtc_ctx.state.playing:
                frame = webrtc_ctx_sendonly.video_receiver.get_frame()

            # Run inference on the frame
            result_class, result_confidence = predict(frame)
            st.session_state.result_class = result_class
            st.session_state.result_confidence = result_confidence

            # Display the frame of the human player
            human_image = np.fliplr(frame.to_ndarray(format="rgb24"))
            human_image_box.image(human_image, use_column_width=True)

            # Randomly pick from rock, paper, scissors
            cpu_pick = random.choice(OBJECTS)
            cpu_image_box.image(IMAGE_PATHS[cpu_pick], use_column_width=True)

            # Victory conditions
            HUMAN_WINS = [
                ("rock", "scissors"),
                ("paper", "rock"),
                ("scissors", "paper"),
            ]

            # Print out some output
            cpu_class_txt.header(f"CPU played {cpu_pick.upper()}")

            if st.session_state.result_class == cpu_pick:
                result_summary.title("It's a tie!")
                rain(emoji="🤝", falling_speed=4, animation_length=1)
            elif (st.session_state.result_class, cpu_pick) in HUMAN_WINS:
                result_summary.title("You win!")
                st.balloons()
            else:
                result_summary.title("Sorry! Better luck next time!")
                rain(emoji="☠️", falling_speed=4, animation_length=1)

            result_class_txt.header(
                f"You played {st.session_state.result_class.upper()}"
            )
            result_confidence_txt.header(
                f"AI confidence: {st.session_state.result_confidence:.2%}"
            )

            my_bar.progress(1.0, text="READY!")
        else:
            # Show this before the play button is pressed
            countdown_images.image(
                "images/rock-paper-scissors.png", use_column_width=True
            )


if __name__ == "__main__":
    main()
