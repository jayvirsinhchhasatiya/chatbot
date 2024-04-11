import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()  # load all the env variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""You are video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 700 words. Please provide the summary of the text given here:  """

def save_uploaded_video(uploaded_video):
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save uploaded video to a temporary file
    video_path = os.path.join(temp_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    return video_path

def transcribe_video(video_path):
    # Extract audio from video
    st.write(video_path)
    video = VideoFileClip(video_path)
    audio = video.audio

    # Save audio to a temporary file
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    # Use SpeechRecognition to transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
    
    return transcript

# getting the summary based on prompt from google gemini pro
def generate_gemini_content_video(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

def get_video_text(uploaded_video):
    summary = ""
    if uploaded_video is not None:
        st.text("Uploaded video:")
        st.video(uploaded_video)

        video_path = save_uploaded_video(uploaded_video)
        
        # if st.button("Transcribe Speech"):
        transcript_text = transcribe_video(video_path)
        # st.write("Transcription:")
        # st.write(transcript)
        summary = generate_gemini_content_video(transcript_text, prompt)
        return summary