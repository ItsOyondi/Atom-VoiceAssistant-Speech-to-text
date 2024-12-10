import streamlit as st
import asyncio
import websockets
import numpy as np
import time
from queue import Queue, Empty
import threading
import librosa
from whisper import load_model
import logging
import soundfile as sf
import os
from sklearn.cluster import KMeans  # Added for KMeans clustering

# Constants
SAMPLE_RATE = 44100
AUDIO_CHUNKS_DIR = "audio_chunks"
FRAME_DURATION = 1  # seconds
CHUNK_DURATION = 10  # seconds
N_MFCC = 13
N_CLUSTERS = 2  # Number of speakers to identify per chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queues for inter-thread communication
audio_queue = Queue()
transcription_queue = Queue()
stop_signal = threading.Event()

# Initialize session state variables
def initialize_session_state():
    if 'server_started' not in st.session_state:
        st.session_state.server_started = False
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []

# WebSocket server for audio capture
async def audio_server(websocket, path=None):
    logger.info("Client connected.")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                try:
                    # Convert the received binary message to audio data
                    audio_data = np.frombuffer(message, dtype=np.float32)
                    audio_queue.put(audio_data)
                    logger.debug(f"Received audio chunk of size: {len(audio_data)}")
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}", exc_info=True)
                    stop_signal.set()
                    break
            else:
                logger.warning("Received non-binary message. Ignoring.")
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client disconnected: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Audio server handler terminating.")

# Main asyncio function to start the WebSocket server
async def websocket_main():
    server = await websockets.serve(audio_server, "0.0.0.0", 8765)
    logger.info("WebSocket server started on ws://0.0.0.0:8765")
    await server.wait_closed()

# Function to start WebSocket server in a thread
def start_audio_server():
    def run_server():
        try:
            asyncio.run(websocket_main())
        except Exception as e:
            logger.error(f"WebSocket server encountered an error: {e}", exc_info=True)
            stop_signal.set()
    
    # Start the WebSocket server in a new daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("WebSocket server thread started.")
    return server_thread

# JavaScript for browser audio capture
def inject_audio_capture_js():
    st.components.v1.html(
        """
        <script>
            (async () => {
                const maxRetries = 5;
                let retries = 0;

                async function connectWebSocket() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        const audioContext = new AudioContext();
                        const mediaStreamSource = audioContext.createMediaStreamSource(stream);
                        const bufferSize = 4096;
                        const numberOfInputChannels = 1;
                        const numberOfOutputChannels = 1;

                        const scriptProcessor = audioContext.createScriptProcessor(bufferSize, numberOfInputChannels, numberOfOutputChannels);

                        mediaStreamSource.connect(scriptProcessor);
                        scriptProcessor.connect(audioContext.destination);

                        const webSocket = new WebSocket('ws://localhost:8765');

                        webSocket.binaryType = 'arraybuffer';

                        webSocket.onopen = () => {
                            console.log("WebSocket connection opened.");
                            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                                if (webSocket.readyState === WebSocket.OPEN) {
                                    const audioData = audioProcessingEvent.inputBuffer.getChannelData(0);
                                    // Clone the buffer to prevent issues with the next onaudioprocess call
                                    const audioBuffer = new Float32Array(audioData);
                                    webSocket.send(audioBuffer.buffer);
                                }
                            };
                        };

                        webSocket.onerror = (error) => {
                            console.error("WebSocket error:", error);
                            webSocket.close();
                        };

                        webSocket.onclose = () => {
                            console.log("WebSocket connection closed.");
                        };
                    } catch (error) {
                        if (retries < maxRetries) {
                            retries++;
                            console.warn(`Retrying WebSocket connection (${retries}/${maxRetries})...`);
                            setTimeout(connectWebSocket, 1000);
                        } else {
                            console.error("Failed to connect to WebSocket server after maximum retries.", error);
                        }
                    }
                }

                connectWebSocket();
            })();
        </script>
        """,
        height=0,
    )

# Capture audio function
def capture_audio():
    all_audio_chunks = []
    while not stop_signal.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.1)
            if len(audio_chunk) > 0:
                # logger.info(f"Captured audio chunk of length: {len(audio_chunk)}")
                all_audio_chunks.append(audio_chunk)
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Error capturing audio: {e}", exc_info=True)
            stop_signal.set()
    
    # Save all the audio chunks at the end
    # if all_audio_chunks:
    #     combined_audio = np.concatenate(all_audio_chunks, axis=0)
    #     try:
    #         os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
    #         output_path = os.path.join(AUDIO_CHUNKS_DIR, "final_audio.wav")
    #         sf.write(output_path, combined_audio, SAMPLE_RATE)
    #         logger.info(f"Saved all captured audio chunks to {output_path}")
    #     except Exception as e:
    #         logger.error(f"Failed to save audio file: {e}", exc_info=True)

# Get transcriptions function with Speaker Diarization
def get_transcriptions():
    """
    Function that processes audio data, performs speaker diarization, and puts transcriptions into the transcription_queue.
    """
    try:
        # Load the Whisper model
        model = load_whisper_model()
        logger.info("Whisper model loaded.")

        buffer = np.array([], dtype=np.float32)
        chunk_size = CHUNK_DURATION * SAMPLE_RATE  # 10 seconds

        while not stop_signal.is_set():
            try:
                # Wait for new audio data with a timeout to allow graceful shutdown
                audio_chunk = audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, audio_chunk))
                logger.debug(f"Buffer length: {len(buffer)} samples.")

                # If buffer has enough samples, process
                while len(buffer) >= chunk_size:
                    # Extract the chunk to transcribe
                    audio_to_transcribe = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]

                    # Perform speaker diarization
                    speaker_transcripts = perform_speaker_diarization(audio_to_transcribe)

                    # Transcribe each speaker's segment
                    for speaker_id, segment_audio in speaker_transcripts:
                        # Resample to 16000 Hz as required by Whisper
                        audio_resampled = librosa.resample(segment_audio, orig_sr=SAMPLE_RATE, target_sr=16000)

                        # Save the segment to a temporary file
                        temp_filename = "temp_audio.wav"
                        sf.write(temp_filename, audio_resampled, 16000)

                        # Perform transcription
                        transcription = model.transcribe(temp_filename, fp16=False)
                        text = transcription.get('text', '').strip()
                        if text:
                            logger.info(f"{speaker_id}: {text}")
                            # Put the speaker label and transcription into the transcription_queue
                            transcription_queue.put((speaker_id, text))

                        # Remove the temporary file
                        try:
                            os.remove(temp_filename)
                        except Exception as e:
                            logger.warning(f"Could not remove temporary file: {e}")

            except Empty:
                continue  # No audio data received, continue waiting
            except Exception as e:
                logger.error(f"Error during transcription: {e}", exc_info=True)
                stop_signal.set()
                break

    except Exception as e:
        logger.error(f"Failed to initialize transcription: {e}", exc_info=True)
        stop_signal.set()

def perform_speaker_diarization(audio_chunk):
  
    # Define frame parameters
    frame_length = FRAME_DURATION * SAMPLE_RATE  # e.g., 1 second frames
    total_length = len(audio_chunk)
    num_frames = total_length // frame_length

    if num_frames == 0:
        return []

    # Split audio into frames
    frames = []
    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frames.append(audio_chunk[start:end])

    # Extract MFCC features for each frame
    mfcc_features = []
    for frame in frames:
        mfcc = extract_features(frame, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        if mfcc is not None:
            mfcc_features.append(mfcc)

    if not mfcc_features:
        return []

    mfcc_features = np.array(mfcc_features)

    # Perform KMeans clustering
    try:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
        kmeans.fit(mfcc_features)
        labels = kmeans.labels_
    except Exception as e:
        logger.error(f"Error during KMeans clustering: {e}", exc_info=True)
        return []

    # Assign speaker IDs based on clustering
    speaker_segments = {}
    for idx, label in enumerate(labels):
        speaker_id = f"Speaker {label + 1}"
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        speaker_segments[speaker_id].append(frames[idx])

    # Concatenate frames per speaker
    speaker_transcripts = []
    for speaker_id, segments in speaker_segments.items():
        speaker_audio = np.concatenate(segments, axis=0)
        speaker_transcripts.append((speaker_id, speaker_audio))

    return speaker_transcripts

# Function to extract MFCC features from an audio chunk
def extract_features(audio_chunk, sr=SAMPLE_RATE, n_mfcc=13):
    
    try:
        # Resample if necessary
        if sr != 16000:
            audio_resampled = librosa.resample(audio_chunk, orig_sr=sr, target_sr=16000)
        else:
            audio_resampled = audio_chunk

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio_resampled, sr=16000, n_mfcc=n_mfcc)
        # Compute the mean MFCCs across time frames
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        logger.error(f"Error extracting features: {e}", exc_info=True)
        return None

# Cached resource for the Whisper model to avoid reloading
@st.cache_resource
def load_whisper_model():
    return load_model("base")

# Main function for Streamlit app
def main():
    st.title("Real-Time Audio Recorder and Transcriber with Speaker Diarization")

    # Initialize session state
    initialize_session_state()

    # Start and Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Recording")
    with col2:
        stop_button = st.button("Stop Recording")

    # Placeholder for transcriptions
    transcription_placeholder = st.empty()

    # Handle Start Recording
    if start_button and not st.session_state.server_started:
        st.session_state.server_started = True
        stop_signal.clear()

        # Start WebSocket server
        start_audio_server()

        # Inject JavaScript for audio capture
        inject_audio_capture_js()

        # Start audio capture thread
        capture_thread = threading.Thread(target=capture_audio, daemon=True)
        capture_thread.start()
        logger.info("Audio capture thread started.")

        # Start transcription thread
        transcription_thread = threading.Thread(target=get_transcriptions, daemon=True)
        transcription_thread.start()
        logger.info("Transcription thread started.")

        st.success("Recording started...")

    # Handle Stop Recording
    if stop_button and st.session_state.server_started:
        stop_signal.set()
        st.session_state.server_started = False

        st.success("Recording stopped.")

    # Real-time transcription updates
    
    while not transcription_queue.empty():
        try:
            speaker_id, transcription = transcription_queue.get_nowait()
            st.session_state.transcriptions.append(f"{speaker_id}: {transcription}")
        except Empty:
            break
        except Exception as e:
            logger.error(f"Error processing transcription from queue: {e}", exc_info=True)

    # Update the transcription display
    if st.session_state.transcriptions:
        transcription_placeholder.text_area(
            "Transcriptions",
            "\n".join(st.session_state.transcriptions),
            height=300
        )

if __name__ == "__main__":
    main()
