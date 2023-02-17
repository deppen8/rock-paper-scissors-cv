# Rock, Paper, Scissors, Computer Vision

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deppen8-rock-paper-scissors-cv-app-jfk0l9.streamlit.app)

This repo includes code for a rock-paper-scissors game that is powered by a computer vision model. The model detects what the user has played and then compares it to a randomly-generated response from a virtual opponent.

The app was built as a demo for elementary school kids to teach them about computer vision/machine learning. The app is hosted on Streamlit Cloud and can be played in the browser. The app can also be run locally.

## Gameplay

When you launch the app, you will need to allow your browser to access to your webcam. After that, click `START` to launch a feed from your webcam.

After your webcam has started...

1. Click `PLAY` to begin the game. The app will start a countdown of "ROCK...", "PAPER...", "SCISSORS...", "SHOOT!".
2. At "SHOOT!", you should play rock, paper, or scissors in view of the camera.
3. The app will use the computer vision model to detect what you have played and generate a random response from the virtual opponent
4. Finally, you will see the results of the game (and the model).

> **Note**
> The model was trained with images that had only a single hand visible in front of either a solid white or solid green background. If you are playing the game with a different setup, you will likely get inaccurate model results.

## Play online

The easiest way to play is via your browser. The app runs as a public Streamlit Cloud app. Click the badge to launch the app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deppen8-rock-paper-scissors-cv-app-jfk0l9.streamlit.app)

> **Warning**
> The app is hosted on Streamlit Cloud, which is a free service. As such, the hardware powering the app is not great and the app can feel sluggish. For best performance, play the game locally.

## Play locally

To play the game locally, you will need to clone the repo and install the dependencies from the `requirements.txt` file. Then you can launch the app with Streamlit.

```bash
cd /my/directory/structure/rock-paper-scissors-cv
streamlit run app.py
```

This will start the app on port 8501. If your browser does not automatically launch a new window or tab, navigate to `http://localhost:8501` in the browser's address bar. You should see the app running there.

## Model training

The models were built using the [Teachable Machine](https://teachablemachine.withgoogle.com/) tool from Google. Because of this, I do not have a ton of visibility into the model training process, but Teachable Machine's documentation says that it uses transfer learning to train from a base mobilenet model.

I used a batch size of 16 and trained for 40 epochs.

The models were exported and included in the repo as Keras models.

### Data

There are two different models available: one with lots of training data and one with little training data. These can be used to show how model performance can be radically different depending on the training data it has seen.

#### "Lots of training data" model

This model was trained with three sources of training images:

1. A few hundred frames from a webcam feed of me and my kids (six and three years old, respectively) playing rock, paper, and scissors.
2. 2,299 frames from the [Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) Kaggle dataset.
3. 2,520 frames from the training set in the [rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) TensorFlow dataset. (This dataset was downloaded from [Roboflow](https://public.roboflow.com/classification/rock-paper-scissors) as the original link from TensorFlow does not seem to work anymore.)

The Teachable Machine project file used for this model, `project.tm`, is available in the repository.

#### "Little training data" model

This model was trained with only one source of training images: 959 frames collected by elementary school students as part of a demonstration about computer vision and machine learning.
