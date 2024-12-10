import assemblyai as aai

aai.settings.api_key = "a004947b59b742e3a1aa9ebc1890f23b"

def aai_model(file_path):
    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=config)

    speaker_transcripts = []
    for utterance in transcript.utterances:
        speaker_transcripts.append(f"Speaker {utterance.speaker}: {utterance.text}")
    return "\n".join(speaker_transcripts)
    
if __name__ == "__main__":
    file_path = "audio_files/output.wav"
    print(aai_model(file_path))