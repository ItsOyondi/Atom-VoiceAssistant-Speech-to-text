

import streamlit as st
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import whisper
from speech_recognition import Microphone, Recognizer
import threading
import queue
import time
# from audio_recorder_streamlit import audio_recorder
import os
import assemblyai as aai
# from aai import TranscriptionConfig, Transcriber  # Ensure this is imported properly
aai.settings.api_key = "a004947b59b742e3a1aa9ebc1890f23b"

# Global variables
audio_queue = queue.Queue()
stop_signal = threading.Event()
transcription_results = []

# Load Whisper model
@st.cache_resource
def load_model(model_name="base"):
    return whisper.load_model(model_name)

# Audio capture function
def capture_audio(sr=16000, duration=5):
    recognizer = Recognizer()
    mic = Microphone(sample_rate=sr)

    while not stop_signal.is_set():
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source, duration=duration)
                # Convert audio to numpy array
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                audio_queue.put(audio_data)
        except Exception as e:
            st.warning(f"Audio capture error: {e}")
            audio_queue.put(None)
            stop_signal.set()


def aai_model(file_path):
    """
    Transcribes audio from a file with speaker diarization.
    """
    aai_client = aai.Client("a004947b59b742e3a1aa9ebc1890f23b")  # Replace with your actual AssemblyAI API key

    # Upload the audio file
    upload_response = aai_client.upload(file_path)
    audio_url = upload_response['upload_url']

    # Start transcription with speaker diarization
    transcript_response = aai_client.transcribe(audio_url, speaker_labels=True)

    # Wait for the transcription to complete
    while transcript_response['status'] not in ['completed', 'failed']:
        transcript_response = aai_client.get_transcription_result(transcript_response['id'])

    if transcript_response['status'] == 'failed':
        raise Exception(f"Transcription failed: {transcript_response.get('error')}")

    # Format transcription result
    speaker_transcripts = []
    for utterance in transcript_response['utterances']:
        speaker_transcripts.append(f"Speaker {utterance['speaker']}: {utterance['text']}")
    return "\n".join(speaker_transcripts)

def process_audio(audio_queue, sr=16000, silence_thresh=0.02, min_silence_duration=0.5):
    """
    Processes audio chunks from a queue, performs speaker diarization and transcription.
    """
    transcription_results = []
    stop_signal = threading.Event()
    accumulated_audio = []
    silence_counter = 0
    chunk_duration = 0.1  # Assuming each audio chunk is ~100ms
    file_path = "temp_audio.wav"

    while not stop_signal.is_set():
        try:
            if audio_queue.empty():
                time.sleep(chunk_duration)
                silence_counter += chunk_duration
                if silence_counter >= min_silence_duration and accumulated_audio:
                    # Save the accumulated audio to a temporary file
                    voiced_audio = np.concatenate(accumulated_audio)
                    librosa.output.write_wav(file_path, voiced_audio, sr=sr)

                    # Transcribe using AAI
                    transcription = aai_model(file_path)
                    transcription_results.append(transcription)

                    # Reset for next speaker
                    accumulated_audio = []
                    silence_counter = 0

                continue

            audio_chunk = audio_queue.get()
            if audio_chunk is None:  # Signal to stop
                break

            # Calculate the energy of the chunk
            energy = np.mean(audio_chunk**2)

            if energy > silence_thresh:
                accumulated_audio.append(audio_chunk)
                silence_counter = 0  # Reset silence counter on voice
            else:
                silence_counter += chunk_duration

                # Check if speaker has stopped
                if silence_counter >= min_silence_duration and accumulated_audio:
                    # Save the accumulated audio to a temporary file
                    voiced_audio = np.concatenate(accumulated_audio)
                    librosa.output.write_wav(file_path, voiced_audio, sr=sr)

                    # Transcribe using AAI
                    transcription = aai_model(file_path)
                    transcription_results.append(transcription)

                    # Reset for next speaker
                    accumulated_audio = []
                    silence_counter = 0

        except Exception as e:
            transcription_results.append(f"Error: {e}")
            stop_signal.set()
            break

    # Clean up temporary file
    if os.path.exists(file_path):
        os.remove(file_path)

    return transcription_results

# Streamlit App
def main():
    st.title("Real-Time Transcription with Whisper")

    # Streamlit session state
    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []

    # Sidebar
    model_name = st.sidebar.selectbox("Whisper Model", ["base", "small", "medium", "large"], index=0)
    start_button = st.sidebar.button("Start Transcription")
    stop_button = st.sidebar.button("Stop Transcription")

    # Start transcription
    if start_button:
        stop_signal.clear()
        st.session_state.transcriptions.clear()
        threading.Thread(target=capture_audio, args=(16000, 5), daemon=True).start()
        threading.Thread(target=process_audio, args=(16000, 15, 2, model_name), daemon=True).start()

    # Stop transcription
    if stop_button:
        stop_signal.set()
        st.warning("Stopping transcription...")

    # Display transcriptions
    st.header("Transcriptions")
    transcription_area = st.empty()

    while not stop_signal.is_set():
        if transcription_results:
            st.session_state.transcriptions.extend(transcription_results)
            transcription_results.clear()
        transcription_area.text("\n".join(st.session_state.transcriptions))
        time.sleep(0.5)

        
if __name__ == "__main__":
    main()