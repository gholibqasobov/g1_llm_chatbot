import time
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel


# ---------------- CONFIG ----------------

MODEL_SIZE = "small"  # small is much faster than medium on CPU
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

SAMPLE_RATE = 16000
CHUNK_SIZE = 512

VAD_THRESHOLD = 0.7        # higher = less noise sensitivity
SILENCE_LIMIT = 1.8        # seconds before stopping speech
MIN_SPEECH_DURATION = 0.4  # ignore short noise bursts

LANGUAGE = "ru"


# ---------------- CLASS ----------------


class SpeechToText:

    def __init__(self):

        print(f"Loading Whisper ({MODEL_SIZE}) on {DEVICE}...")
        self.whisper = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )

        self.audio_buffer = []
        self.is_speaking = False
        self.silence_chunks = 0

        self.max_silence_chunks = int(
            SILENCE_LIMIT / (CHUNK_SIZE / SAMPLE_RATE)
        )

        self.min_speech_chunks = int(
            MIN_SPEECH_DURATION / (CHUNK_SIZE / SAMPLE_RATE)
        )

        self.result_text = None

    # ---------------- TRANSCRIBE ----------------

    def transcribe(self, audio_data):

        segments, _ = self.whisper.transcribe(
            audio_data,
            beam_size=5,
            language=LANGUAGE
        )

        text = " ".join([s.text for s in segments]).strip()

        if text:
            self.result_text = text

    # ---------------- AUDIO CALLBACK ----------------

    def audio_callback(self, indata, frames, time_info, status):

        chunk = indata.flatten()
        chunk_tensor = torch.from_numpy(chunk).float()

        with torch.no_grad():
            speech_prob = self.vad_model(chunk_tensor, SAMPLE_RATE).item()

        # ----- Speech detected -----
        if speech_prob > VAD_THRESHOLD:

            if not self.is_speaking:
                self.is_speaking = True

            self.audio_buffer.append(chunk)
            self.silence_chunks = 0

        # ----- Silence -----
        elif self.is_speaking:

            self.audio_buffer.append(chunk)
            self.silence_chunks += 1

            if self.silence_chunks > self.max_silence_chunks:

                if len(self.audio_buffer) > self.min_speech_chunks:

                    full_audio = np.concatenate(self.audio_buffer)

                    print("Processing speech...")

                    self.transcribe(full_audio)

                # reset state
                self.audio_buffer = []
                self.is_speaking = False
                self.silence_chunks = 0

    # ---------------- LISTEN FUNCTION ----------------

    def listen(self, timeout=5):

        """
        Listen for speech for a limited time window.

        timeout: maximum listening time (seconds)
        """

        print("Listening...")

        self.result_text = None
        start_time = time.time()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
        ):

            while self.result_text is None:

                if time.time() - start_time > timeout:
                    print("Listening timeout.")
                    break

                sd.sleep(100)

        return self.result_text


# ---------------- TEST ----------------

if __name__ == "__main__":

    stt = SpeechToText()

    while True:

        text = stt.listen(timeout=6)

        if text:
            print("\nTranscribed:", text)
        else:
            print("No speech detected.")