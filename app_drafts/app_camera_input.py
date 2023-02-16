import random
import time

import numpy as np
import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
from streamlit.runtime.uploaded_file_manager import UploadedFile

# # Disable scientific notation for clarity
np.set_printoptions(suppress=True)

st.set_page_config(layout="wide")


def main():
    st.title("Rock Paper Scissors")

    if "keras_model" in st.session_state:
        model = st.session_state["keras_model"]
    else:
        # Load the model
        model = load_model("./models/model_v0/keras_Model.h5", compile=False)
        st.session_state["keras_model"] = model

    if "class_names" in st.session_state:
        class_names = st.session_state["class_names"]
    else:
        # Load the class labels
        class_names = open("./models/model_v0/labels.txt", "r").readlines()
        st.session_state["class_names"] = class_names

    if "result_class" in st.session_state:
        result_class = st.session_state["result_class"]
    else:
        result_class = None
        st.session_state["result_class"] = result_class

    if "result_confidence" in st.session_state:
        result_confidence = st.session_state["result_confidence"]
    else:
        result_confidence: float = None
        st.session_state["result_confidence"] = result_confidence

    def predict() -> None:
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
            image_array = np.fliplr(np.asarray(image))

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            st.session_state["result_class"] = f"{class_name[2:]}".rstrip()
            st.session_state["result_confidence"] = float(confidence_score)
        else:
            st.session_state["result"] = "skipped"

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.camera_input(
            "webcam_capture",
            key="webcam_image",
            on_change=predict,
            label_visibility="collapsed",
        )
        result_class_txt = st.header("")
        result_confidence_txt = st.header("")

    with col2:
        cpu_image_box = st.empty()
        if isinstance(st.session_state.webcam_image, UploadedFile):
            image_paths = {
                "rock": "images/rock.png",
                "paper": "images/paper.png",
                "scissors": "images/scissors.png",
            }

            progress_state = 1.0
            my_bar = st.progress(progress_state, text="ROCK...")
            cpu_image_box.image("images/rock.png", width=350)
            for text in ["PAPER...", "SCISSORS..."]:
                time.sleep(0.75)
                progress_state -= 1 / 3
                my_bar.progress(progress_state, text=text)
                cpu_image_box.image(image_paths[text[:-3].lower()], width=350)

            time.sleep(0.75)
            my_bar.progress(0, text="SHOOT!")

            # randomly pick from rock, paper, scissors
            cpu_pick = random.choice(list(image_paths))
            cpu_image_box.image(image_paths[cpu_pick], width=350)

            human_wins = [
                ("rock", "scissors"),
                ("paper", "rock"),
                ("scissors", "paper"),
            ]

            if st.session_state.result_class == cpu_pick:
                st.header("It's a tie!")
            elif (st.session_state.result_class, cpu_pick) in human_wins:
                st.header("You win!")
                st.balloons()
            else:
                st.header("Sorry! Better luck next time!")
                st.snow()

            result_class_txt.header(
                f"Detected: {st.session_state.result_class.upper()}"
            )
            result_confidence_txt.header(
                f"Confidence: {st.session_state.result_confidence:.2%}"
            )
        else:
            cpu_image_box.image("images/rock-paper-scissors.png", width=500)


if __name__ == "__main__":
    main()
