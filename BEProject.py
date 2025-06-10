import streamlit as st
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import time
import json
import logging
import typing_extensions as typing
import os
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
from elevenlabs.client import ElevenLabs
import io
import asyncio
from pydub import AudioSegment
from googletrans import Translator
from spleeter.separator import Separator
import string
import datetime
import base64
import random
import yt_dlp
# Setup logging
logging.basicConfig(level=logging.INFO)

# ElevenLabs API setup (replace with your key)
elevenlabs_client = ElevenLabs(api_key="sk_3261c348d83d0435936ca53fee71cfa27957977ddc599114") # Replace with your actual key.
#translator = Translator()

# Gemini API Key Configuration
def configure_gemini_api(api_key):
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = api_key
        genai.configure(api_key=api_key)

# Video Upload
def upload_video(video_source):
    local_video_path = None
    try:
        if isinstance(video_source, str) and video_source.startswith("http"):  # It's a URL string
            status_text.text("Downloading YouTube video...")
            local_video_path = download_youtube_video(video_source)
            if not local_video_path:
                st.error("Failed to download YouTube video.")
                return None, None
            # Don't remove the downloaded file yet, needed for processing

        else: # It's an UploadedFile object
            status_text.text("Saving uploaded file temporarily...")
            # Save the uploaded file temporarily to get a path
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_source.getvalue())
                local_video_path = temp_file.name
            # Keep local_video_path, it's needed for processing

        # Now upload the local file (downloaded or temporary) to Gemini
        video_file = genai.upload_file(path=local_video_path)
        return video_file, local_video_path # Return Gemini file and the *local* path

    except Exception as e:
        st.error(f"Error during video upload/processing: {e}")
        # Clean up local file if created and an error occurred *before* returning
        if local_video_path and os.path.exists(local_video_path):
             # Be careful here, might need the path for debugging
             # Consider logging the error before removing
             logging.error(f"Cleaning up {local_video_path} after error: {e}")
             # os.remove(local_video_path) # Decide if you want to auto-delete on error
        return None, None
    """ if video_source.startswith("http"):  # It's a URL
        video_path = download_youtube_video(video_source)
        progress_bar.progress(12)
        if video_path:
            video_file = genai.upload_file(path=video_path)
            return video_file, video_path  
        else:
            return None, None
        
    ###
    else:  # It's a local file path (from file uploader)
        video_file = genai.upload_file(path=video_source)
        return video_file, video_source"""
    
    #video_file = genai.upload_file(path=video_file_name)
    #return video_file

# File Activation Check
def wait_for_file_activation(video_file):
    max_wait_time = 300
    start_time = time.time()

    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
        if time.time() - start_time > max_wait_time:
            st.error("File processing took too long.")
            return False

    if video_file.state.name == "FAILED":
        st.error("File processing failed.")
        return False
    elif video_file.state.name == "ACTIVE":
        return True
    else:
        st.error(f"Unexpected file state: {video_file.state.name}")
        return False

# Timestamp Generation
def generate_ball_start_timestamps(video_file):
    class Timestamps(typing.TypedDict):
        timestamp: str
    model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})
    prompt = (
        "Extract the start timestamps of every ball bowled in this cricket video.Do not miss a single second of the video"
        "Provide the timestamps in chronological order, ensuring that no ball is missed and formatted in JSON "
        "as an array named 'ball_timestamps,' with each timestamp a string in the format 'minutes:seconds.' "
        "Do not include any other information or commentary."
    )
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
            return response.text
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                st.error("Failed to generate timestamps after multiple attempts.")
                return None

# Commentary Generation
def generate_commentary_with_timestamps(video_file, ball_timestamps,language):
    time.sleep(60)
    class CommentarySchema(typing.TypedDict):
        timestamp: str
        commentary: str
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20", generation_config={"response_mime_type": "application/json", "response_schema": list[CommentarySchema]})
    ball_timestamps_dict = {"ball_timestamps": [{"ball": i + 1, "timestamp": ts} for i, ts in enumerate(ball_timestamps)]}
    prompt = (f"""You are a professional cricket commentator on air in a live match.
        You are given:
        1. A cricket match video.
        2. Start timestamps for each ball (in MM:SS format).
        Your task:
        - For each ball timestamp given to you ,return the timestamp as it is in 'minutes:seconds' format along with it's corresponding commentary like a professional cricket commentator would do.
        - The commentary should be of appropriate length so that it can be spoken from the start of the current ball to the start of the next ball.
        - You may refer to the audio to understand the video but your commentary should not be the exact same.
        - The language of the commentary should be {language}.
        - Strictly use only the timestamps given to you ,don't generate new timestamps.    
        - There should be only a single commentary associated with each timestamp , not multiple commentaries for the same timestamp.
        - Here is a sample entry of the json {{"commentary": "Oh! Bold him! What a beauty!", "timestamp": "00:03"}} 
        """) # Include your detailed prompt here.
    retries = 3
    for attempt in range(retries):
        try:
            commentary_response = model.generate_content([video_file, prompt, json.dumps(ball_timestamps_dict)], request_options={"timeout": 600})
            return commentary_response.text
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                st.error("Failed to generate commentary after multiple attempts.")
                return None

def convert_timestamp_to_seconds(timestamp_str):
    minutes, seconds = map(int, timestamp_str.split(':'))
    return minutes * 60 + seconds



async def translate_text(text, target_lang):
    translator = Translator()
    translated = await translator.translate(text, src="en", dest=target_lang)
    return translated.text

def sync_translate(text, target_lang):
    return asyncio.run(translate_text(text, target_lang))

def generate_audio_commentary(commentary_data,background_audio_path):
    audio_files = []

    background_audio = AudioFileClip(background_audio_path)

    for i, entry in enumerate(commentary_data):
        timestamp = entry["timestamp"]
        commentary = entry["commentary"]

        temp_commentary_file = f"temp_commentary_{timestamp.replace(':', '_')}.wav"

        try:
            # Generate AI speech using ElevenLabs
            audio = elevenlabs_client.text_to_speech.convert(
                text=commentary,
                voice_id="fRV3NpXPa1DGVnFs6Dg5",
                model_id="eleven_multilingual_v2",
                output_format="pcm_16000",
            )
            audio_data = b"".join(audio)
            pcm_audio = io.BytesIO(audio_data)

            # Convert raw PCM to WAV
            sound = AudioSegment(
                pcm_audio.read(), sample_width=2, frame_rate=16000, channels=1
            )
            sound.export(temp_commentary_file, format="wav")

            # Load generated commentary audio
            commentary_audio = AudioFileClip(temp_commentary_file)

            # Determine duration for background segment
            start_time = convert_timestamp_to_seconds(timestamp)
            if i < len(commentary_data) - 1:
                end_time = convert_timestamp_to_seconds(commentary_data[i + 1]["timestamp"])
                duration = end_time - start_time
            else:
                duration = background_audio.duration - start_time  # Edge case for last entry

            # Extract background segment
            background_segment = background_audio.subclip(start_time, start_time + duration)

            # Mix AI voice with background audio
            mixed_audio = CompositeAudioClip([
                background_segment.set_start(0), 
                commentary_audio.set_start(1.8)
            ])
            
            # Save final mixed audio
            temp_mixed_file = f"temp_mixed_{timestamp.replace(':', '_')}.mp3"
            mixed_audio.write_audiofile(temp_mixed_file, codec="libmp3lame", bitrate="128k", fps=44100)
            mixed_audio.close()
            #commentary_audio.close()
            #background_segment.close()
            # Store file path
            audio_files.append((timestamp, temp_mixed_file))

            # Cleanup temporary files
            os.remove(temp_commentary_file)

        except Exception as e:
            print(f"Error generating audio: {e}")
            return []  # Handle errors
    background_audio.close()
    return audio_files

def separate_audio(video_path):
    """Separates audio into vocals and accompaniment using Spleeter."""
    temp_audio_file = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_file)

    separator = Separator('spleeter:2stems')
    separator.separate_to_file(temp_audio_file, 'separated_audio')

    accompaniment_path = os.path.join('separated_audio', 'temp_audio', 'accompaniment.wav') 
    os.remove(temp_audio_file) 
    os.remove(os.path.join('separated_audio', 'temp_audio', 'vocals.wav')) 

    return accompaniment_path

def merge_audio_with_video(video_path, audio_files, commentary_data):
    video = VideoFileClip(video_path)
    video_duration = video.duration

    audio_clips = []
    for i, (timestamp, audio_file) in enumerate(audio_files):

        minutes, seconds = map(int, timestamp.split(":"))
        start_time = minutes * 60 + seconds


        if start_time > video_duration:
            continue

        if i + 1 < len(audio_files):
            next_timestamp = audio_files[i + 1][0]
            next_minutes, next_seconds = map(int, next_timestamp.split(":"))
            end_time = (next_minutes * 60 + next_seconds) - 0.2
            end_time = min(end_time, video_duration)
            audio_clip = AudioFileClip(audio_file).set_start(start_time).set_duration(end_time - start_time)
        else:
            audio_clip = AudioFileClip(audio_file).set_start(start_time)
        
        audio_clips.append(audio_clip)


    combined_audio = CompositeAudioClip(audio_clips)


    final_video = video.set_audio(combined_audio)


    output_video_path = "final_cricket_commentary_video.mp4"
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


    for _, audio_file in audio_files:
        try:
            os.remove(audio_file)
        except PermissionError as e:
            print(f"Warning: Could not delete temporary file {audio_file}: {e}")
            #  You could log this, but since the video is created, it's safe to ignore.
        except OSError as e: #Catch other OS errors
            print(f"Warning: Could not delete temporary file {audio_file}: {e}")

    return output_video_path

def create_vtt_subtitles(parsed_commentary, vtt_filename="subtitles.vtt", fixed_duration=3):
    """Generates a VTT file from translated commentary (mm:ss format)"""
    with open(vtt_filename, "w", encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")  # VTT file header

        for i in range(len(parsed_commentary)):
            start_time = parsed_commentary[i]["timestamp"]  # mm:ss format
            commentary = parsed_commentary[i]["commentary"]

            # Determine the end timestamp
            if i + 1 < len(parsed_commentary):
                end_time = parsed_commentary[i + 1]["timestamp"]  # Use next start as end time
            else:
                end_time = calculate_end_time(start_time, fixed_duration)  # Add a fixed duration for the last one

            # Convert timestamps to WebVTT format
            start_time_vtt = convert_to_vtt_format(start_time)
            end_time_vtt = convert_to_vtt_format(end_time)

            vtt_file.write(f"{start_time_vtt} --> {end_time_vtt}\n{commentary}\n\n")

    return vtt_filename

def convert_to_vtt_format(timestamp):
    """Converts 'mm:ss' format to '00:mm:ss.000' format required for VTT"""
    minutes, seconds = map(int, timestamp.split(":"))
    return f"00:{minutes:02}:{seconds:02}.000"

def calculate_end_time(start_time, duration=3):
    """Adds a fixed duration (default 3s) to the timestamp and returns in mm:ss format"""
    minutes, seconds = map(int, start_time.split(":"))
    total_seconds = minutes * 60 + seconds + duration
    new_minutes, new_seconds = divmod(total_seconds, 60)
    return f"{new_minutes:02}:{new_seconds:02}"


def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: left;
        background-repeat: no-repeat;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


def create_background_audio_video(video_path, background_audio_path, output_path="background_audio_video.mp4"):
    """Creates a video with only the background audio."""
    try:
        video_clip = VideoFileClip(video_path)
        background_audio_clip = AudioFileClip(background_audio_path)

        # Ensure the background audio is at least as long as the video
        if background_audio_clip.duration < video_clip.duration:
            background_audio_clip = background_audio_clip.audio_loop(duration=video_clip.duration)
        
        # Set the video's audio to the background audio, trimmed to video length
        video_with_background = video_clip.set_audio(background_audio_clip.subclip(0, video_clip.duration))

        video_with_background.write_videofile(output_path, codec="libx264", audio_codec="aac") # Changed Codec from libx264 to libx64
        return output_path

    except Exception as e:
        print(f"Error creating background audio video: {e}")
        return None
    
def download_youtube_video(url):
    """Downloads a YouTube video using yt-dlp."""
    try:
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        filename_template = f'temp_yt_video_{random_id}.%(ext)s'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Get best mp4
            'outtmpl': filename_template,  # Output template
            'noplaylist': True,  # Prevent downloading entire playlists
            'quiet': True, # Suppress output
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict) #get filename from info_dict
            return video_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None
    
# Streamlit App
st.set_page_config(page_title="Automated Cricket Commentary")

# Set Background (Replace with your local image path)
set_background("image.webp")

st.title("Automated Cricket Commentary")

progress_bar = st.progress(0)
status_text = st.empty()

first_api_key = 'AIzaSyBswOmN5Qu99vTf2l_pmX0ot4iCkxMP920'  # Replace with your first API key
configure_gemini_api(first_api_key)
#language_options = {"English": "en", "Hindi": "hi"}
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh-TW",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Dutch": "nl",
    "Turkish": "tr",
    "Swedish": "sv",
    "Greek": "el",
    "Polish": "pl",
    "Thai": "th",
    "Vietnamese": "vi",
    "Hebrew": "he",
    "Danish": "da",
    "Finnish": "fi",
    "Czech": "cs",
    "Hungarian": "hu",
    "Norwegian": "no",
    "Indonesian": "id"
}

selected_language = st.selectbox("Select Language:", list(language_options.keys()), index=0)
selected_language_code = language_options[selected_language]

youtube_url = st.text_input("Enter YouTube URL:", "")
uploaded_file = st.file_uploader("Upload a cricket video", type=["mp4"])

if youtube_url or uploaded_file:
    # Use the YouTube URL if provided, otherwise use the uploaded file
    video_source = youtube_url if youtube_url else uploaded_file

    if video_source:  # Proceed only if a source is provided
        if isinstance(video_source, str): #if the video_source is a youtube url string
             status_text.text("Downloading video...")
        else: #if its a local file
             status_text.text("Uploading video...")
        video_file, video_path = upload_video(video_source)

    if video_file and wait_for_file_activation(video_file):
        #status_text.text("Video Uploaded")
        #progress_bar.progress(33)

        timestamps_json = generate_ball_start_timestamps(video_file)

        if timestamps_json:
            try:
                parsed_response = json.loads(timestamps_json)
                if isinstance(parsed_response, dict):
                    ball_timestamps = [timestamp for timestamp in parsed_response.values()]

                elif isinstance(parsed_response, list):
                    ball_timestamps = parsed_response 
                else:
                    st.error("Unexpected response format.")
                    ball_timestamps = []
                status_text.text("Video Uploaded")
                progress_bar.progress(25)
                commentary_json = generate_commentary_with_timestamps(video_file, ball_timestamps,selected_language)
                status_text.text("Commentary Generated")
                progress_bar.progress(50)
                if commentary_json:
                    try:
                        parsed_commentary = json.loads(commentary_json)
                        if isinstance(parsed_commentary, list) and all(isinstance(entry, dict) and "commentary" in entry and "timestamp" in entry for entry in parsed_commentary):

                            #st.subheader("Generated Commentary:")
                            #if selected_language_code != "en":
                                #for entry in parsed_commentary:
                                #st.write(f"**{entry['timestamp']}**: {entry['commentary']}")
                                    #entry["commentary"]=sync_translate(entry["commentary"],selected_language_code)
                            vtt_file = create_vtt_subtitles(parsed_commentary)
                            try:
                                background_audio = separate_audio(video_path)
                                background_video_path = create_background_audio_video(video_path, background_audio)
                                if background_video_path:
                                    st.write("Original Video with Background Audio:")
                                    st.video(background_video_path)
                                audio_files = generate_audio_commentary(parsed_commentary,background_audio)
                                status_text.text("Audio Generated")
                                progress_bar.progress(75)
                                final_video = merge_audio_with_video(video_path, audio_files, parsed_commentary)
                                st.write("AI Generated Commentary Video")
                                st.video(final_video,subtitles=vtt_file)
                                status_text.text("Video Ready")
                                progress_bar.progress(100)
                                #status_text.text("Commentary generated and audio added successfully!")

                            except Exception as e:
                                st.error(f"Error processing audio or video: {e}") # More helpful error messages

                        # ... (rest of error handling same as before)
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding JSON response for commentary: {e}")  # More helpful error messages

            except json.JSONDecodeError as e:
                 st.error(f"Error decoding JSON response for timestamps: {e}")  # More helpful error messages

