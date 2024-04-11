import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()  # load all the env variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 700 words. Please provide the summary of the text given here:  """

# getting transcript data from yt videos 
def extract_transcript_details(youtube_url):
    try:
        youtube_id = youtube_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(youtube_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    except Exception as e:
        raise e
    
# getting the summary based on prompt from google gemini pro
def generate_gemini_content_youtube(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text


def get_yt_summary(youtube_link):
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get detailed notes"):
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content_youtube(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)


def get_yt_summary_test(youtube_link):
    summary = ""
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        transcript_text = extract_transcript_details(youtube_link)

        summary = generate_gemini_content_youtube(transcript_text, prompt)

    return summary        

