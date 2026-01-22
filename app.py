import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Music Mood Generator")

st.title("🎵 AI Music Remix & Mood Generator")
st.write("Generate music using AI based on your mood or genre")

# ---------------- User Inputs ----------------
mood = st.selectbox(
    "Select Mood / Genre",
    ["Happy", "Sad", "Relaxing", "EDM", "Lo-Fi"]
)

duration = st.slider("Music Duration (seconds)", 5, 30, 10)

# ---------------- Load AI Model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-to-audio",
        model="facebook/musicgen-small"
    )

generator = load_model()

# ---------------- Music Generation Function ----------------
def generate_music(mood, duration):
    prompt = f"{mood} music"
    audio = generator(
        prompt,
        forward_params={"max_new_tokens": duration * 50}
    )

    os.makedirs("generated_music", exist_ok=True)
    file_path = f"generated_music/{mood}.wav"

    wavfile.write(file_path, 32000, audio["audio"])
    return file_path

# ---------------- Button Action ----------------
if st.button("Generate Music"):
    st.info("🎶 AI is generating music... please wait")

    music_file = generate_music(mood, duration)

    st.success("Music Generated Successfully!")
    st.audio(music_file)

    st.download_button(
        "Download Music",
        open(music_file, "rb"),
        file_name="ai_music.wav"
    )
