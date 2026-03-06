import torch
import numpy as np
import sounddevice as sd
import os
import sys
import re
import importlib.util
from pathlib import Path


# ------------------------------------------------------------------
#  Module-level helper for SSML generation
# ------------------------------------------------------------------

def _ssml_add_pauses(sentence):
    """Stretch the word before an em-dash and add a brief pause after it.

    "Я — Жорик" → '<prosody rate="slow">Я+</prosody> <break time="150ms"/> Жорик'
    """
    def _repl(m):
        word = m.group(1)
        if '+' not in word:
            word = _add_stress(word)
        return f'<prosody rate="slow">{word}</prosody> <break time="150ms"/> '

    # word + whitespace + em-dash/en-dash/double-dash + whitespace
    sentence = re.sub(r'(\S+)\s+[—–]\s+', _repl, sentence)
    sentence = re.sub(r'(\S+)\s+--\s+', _repl, sentence)
    return sentence


def _ssml_stretch_question(sentence):
    """
    For questions, slow down the last word before '?' to create a natural
    rising, drawn-out question intonation — the way humans actually ask.

    Example:
      "Как дела?" → "Как <prosody rate="slow">дела+?</prosody>"
    """
    # Match the last word (possibly with a stress mark +) right before ?
    m = re.match(r'^(.*\s)(\S+?)(\?+)$', sentence, re.DOTALL)
    if m:
        prefix, last_word, qmark = m.groups()
        # Add stress mark (+) before the last vowel if not already present
        if '+' not in last_word:
            last_word = _add_stress(last_word)
        return f'{prefix}<prosody rate="slow">{last_word}{qmark}</prosody>'
    return sentence


def _add_stress(word):
    """Add Silero stress mark (+) after the last vowel in a word."""
    vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
    # Find the last vowel position
    last_vowel_idx = -1
    for i, ch in enumerate(word):
        if ch in vowels:
            last_vowel_idx = i
    if last_vowel_idx >= 0:
        return word[:last_vowel_idx + 1] + '+' + word[last_vowel_idx + 1:]
    return word


class TextToSpeech:
    """Converts text to speech using Silero TTS (local model, no internet required)."""

    def __init__(
        self,
        language="ru",
        model_id="v5_ru",
        speaker="eugene",
        sample_rate=48000,
        device=None,
        put_accent=True,
        put_yo=True,
        put_stress_homo=True,
        put_yo_homo=True,
    ):
        """
        Initialize the Text-to-Speech processor.

        Args:
            language: Language code (e.g., "ru")
            model_id: Model identifier (e.g., "v5_ru")
            speaker: Speaker voice name
            sample_rate: Audio sample rate
            device: torch device (auto-detect if None)
            put_accent: Add accent to text
            put_yo: Replace ё in text
            put_stress_homo: Add stress marks
            put_yo_homo: Handle ё homonyms
        """
        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.put_stress_homo = put_stress_homo
        self.put_yo_homo = put_yo_homo

        # Set device — default to CPU to preserve GPU VRAM for STT.
        # Silero TTS is lightweight and runs with negligible latency on CPU.
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # Lazy load - model not loaded yet
        self.model = None
        print(f"Text-to-Speech initialized (device={self.device}, model will load on first use).\n")

    def _load_model(self):
        """Load the Silero TTS model from local cache (no internet required)."""
        if self.model is not None:
            return  # Already loaded

        print(f"Loading Silero TTS model (language={self.language}, model={self.model_id})...")
        
        cache_dir = Path.home() / '.cache/torch/hub/snakers4_silero-models_master'
        
        if not cache_dir.exists():
            raise RuntimeError(
                f"\n❌ Silero TTS model not found in local cache at:\n"
                f"   {cache_dir}\n\n"
                f"Please download it first using:\n"
                f"   python utils/download_tts_model.py\n"
            )
        
        try:
            # Save current working directory
            original_cwd = os.getcwd()
            
            try:
                # Change to cache directory so relative imports work
                os.chdir(str(cache_dir))
                
                # Save and modify sys.path: remove entries that contain our
                # project's 'src' package so Python finds the Silero cache's
                # src/ instead.  The Silero cache dir must be first on the path.
                original_path = sys.path.copy()
                project_root = str(Path(__file__).resolve().parents[2])
                sys.path = [str(cache_dir)] + [
                    p for p in sys.path
                    if p != project_root and p != str(cache_dir)
                ]
                
                # Temporarily remove conflicting 'src' entries from sys.modules
                # so that hubconf's `from src.silero import ...` resolves to
                # the src/ inside the Silero cache, not our project's src/.
                saved_modules = {}
                for mod_name in list(sys.modules.keys()):
                    if mod_name == 'src' or mod_name.startswith('src.'):
                        saved_modules[mod_name] = sys.modules.pop(mod_name)

                # Ensure the Silero cache's src/ is importable as a package
                # (it lacks __init__.py, so we create a temporary one in-memory)
                import types
                src_pkg = types.ModuleType('src')
                src_pkg.__path__ = [str(cache_dir / 'src')]
                sys.modules['src'] = src_pkg
                
                # Load hubconf module
                hubconf_path = cache_dir / 'hubconf.py'
                spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
                hubconf = importlib.util.module_from_spec(spec)
                
                # Execute the hubconf module
                spec.loader.exec_module(hubconf)
                
                # Call the silero_tts function from hubconf
                result = hubconf.silero_tts(
                    language=self.language,
                    speaker=self.model_id
                )
                
                # Unpack the result
                if isinstance(result, tuple):
                    self.model = result[0]
                else:
                    self.model = result
                
                self.model.to(self.device)
                print(f"Text-to-Speech model loaded. Available speakers: {self.model.speakers}\n")
                
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
                
                # Restore original sys.path
                sys.path = original_path
                
                # Remove Silero's src modules and restore our project's src modules
                for mod_name in list(sys.modules.keys()):
                    if mod_name == 'src' or mod_name.startswith('src.'):
                        sys.modules.pop(mod_name, None)
                sys.modules.update(saved_modules)
            
            return
            
        except Exception as e:
            raise RuntimeError(
                f"\n❌ Error loading Silero TTS model: {e}\n"
                f"Try re-downloading the model:\n"
                f"   python utils/download_tts_model.py\n"
            )

    def text_to_speech(self, text, speaker=None, play=True):
        """
        Convert text to speech and optionally play it.

        Args:
            text: Text to convert to speech
            speaker: Speaker voice (uses default if None)
            play: If True, play audio immediately

        Returns:
            audio: numpy array of audio samples
        """
        # Load model on first use
        self._load_model()

        if speaker is None:
            speaker = self.speaker

        audio = self.model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=self.sample_rate,
            put_accent=self.put_accent,
            put_yo=self.put_yo,
            put_stress_homo=self.put_stress_homo,
            put_yo_homo=self.put_yo_homo,
        )

        if play:
            self.play_audio(audio)

        return audio

    def text_to_speech_ssml(self, ssml_text, speaker=None, play=True):
        """
        Convert SSML-formatted text to speech with expressive control.

        Supports Silero SSML tags for natural, emotional speech:
          - <prosody rate="..."> : x-slow, slow, medium, fast, x-fast
          - <prosody pitch="...">: x-low, low, medium, high, x-high
          - <break time="500ms"/>: pauses of arbitrary length
          - <p>...</p>           : paragraph breaks
          - <s>...</s>           : sentence breaks

        The ssml_text should be wrapped in <speak>...</speak> tags.
        If it isn't, it will be wrapped automatically.

        Example::

            ssml = '''
            <speak>
              <p>
                Привет! <prosody rate="slow">Как дела?</prosody>
                <break time="500ms"/>
                <prosody pitch="high">Отлично!</prosody>
              </p>
            </speak>
            '''
            tts.text_to_speech_ssml(ssml)

        Args:
            ssml_text: SSML-formatted text string
            speaker: Speaker voice (uses default if None)
            play: If True, play audio immediately

        Returns:
            audio: Audio tensor
        """
        self._load_model()

        if speaker is None:
            speaker = self.speaker

        # Auto-wrap in <speak> if the user forgot
        stripped = ssml_text.strip()
        if not stripped.startswith("<speak>"):
            ssml_text = f"<speak>{stripped}</speak>"

        audio = self.model.apply_tts(
            ssml_text=ssml_text,
            speaker=speaker,
            sample_rate=self.sample_rate,
        )

        if play:
            self.play_audio(audio)

        return audio

    # ------------------------------------------------------------------
    #  Natural speech: auto-convert plain text → expressive SSML
    # ------------------------------------------------------------------

    @staticmethod
    def text_to_natural_ssml(text):
        """
        Convert plain text into SSML with gentle, human-like prosody.

        Philosophy: Silero already handles intonation from punctuation well.
        We only use SSML to provide structural hints (<s>, <p>) and apply
        subtle prosody nudges for clearly emotional punctuation (?, !, ...).
        Most sentences are left at default prosody so Silero's own natural
        intonation shines through.

        Args:
            text: Plain text string (LLM response)

        Returns:
            str: SSML string ready for text_to_speech_ssml()
        """
        # Clean up whitespace / markdown artefacts the LLM may produce
        text = text.strip()
        text = re.sub(r'\*+', '', text)           # strip bold/italic markers
        text = re.sub(r'#{1,6}\s*', '', text)     # strip markdown headings
        text = re.sub(r'\n{2,}', '\n', text)      # collapse multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)        # collapse whitespace

        # Split into paragraphs (by newline)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        ssml_parts = ['<speak>']

        for para in paragraphs:
            ssml_parts.append('  <p>')

            # Split paragraph into sentences (keep the delimiter)
            sentences = re.split(r'(?<=[.!?…])\s+', para)

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                # Apply em-dash pauses
                sent_with_pauses = _ssml_add_pauses(sent)

                # Only apply prosody for strongly emotional punctuation.
                # Normal sentences get no prosody wrapper.
                if sent.endswith('?'):
                    # Questions: stretch the last word for natural rising tone
                    ssml_sent = f'    <s>{_ssml_stretch_question(sent_with_pauses)}</s>'
                elif sent.endswith('!'):
                    # Exclamations: gentle pitch lift
                    ssml_sent = (
                        f'    <s><prosody pitch="high">'
                        f'{sent_with_pauses}'
                        f'</prosody></s>'
                    )
                elif sent.endswith('…') or sent.endswith('...'):
                    # Trailing off: slightly slower for dramatic effect
                    ssml_sent = (
                        f'    <s><prosody rate="slow">'
                        f'{sent_with_pauses}'
                        f'</prosody></s>'
                    )
                else:
                    # Normal statements and questions — let Silero handle
                    # intonation naturally from punctuation alone
                    ssml_sent = f'    <s>{sent_with_pauses}</s>'

                ssml_parts.append(ssml_sent)

            ssml_parts.append('  </p>')

        ssml_parts.append('</speak>')
        return '\n'.join(ssml_parts)

    def speak_natural(self, text, speaker=None, play=True):
        """
        Speak plain text with natural, expressive prosody.

        Convenience wrapper: converts text → SSML → audio.

        Args:
            text: Plain text to speak
            speaker: Speaker voice (uses default if None)
            play: If True, play audio immediately

        Returns:
            audio: Audio tensor
        """
        ssml = self.text_to_natural_ssml(text)
        return self.text_to_speech_ssml(ssml, speaker=speaker, play=play)


    def play_audio(self, audio):
        """
        Play audio samples (blocking, non-interruptible).

        Args:
            audio: Audio tensor or numpy array
        """
        sd.play(audio, self.sample_rate)
        sd.wait()

    def play_audio_interruptible(self, audio, should_stop_fn):
        """
        Play audio in small chunks, checking should_stop_fn() between chunks.

        This enables barge-in: if the user starts speaking during playback,
        the caller's should_stop_fn returns True and playback stops.

        Args:
            audio: Audio tensor or numpy array
            should_stop_fn: Callable returning True to interrupt playback

        Returns:
            bool: True if playback was interrupted, False if completed
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        chunk_size = self.sample_rate // 10  # 100 ms chunks

        with sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32"
        ) as stream:
            for i in range(0, len(audio_np), chunk_size):
                if should_stop_fn():
                    return True
                chunk = audio_np[i : i + chunk_size]
                stream.write(chunk.reshape(-1, 1))

        return False

    def unload(self):
        """Unload the TTS model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print("TTS model unloaded.")

    def get_speakers(self):
        """Get available speakers for current model."""
        self._load_model()
        return self.model.speakers


def main():
    """Test the Text-to-Speech module."""
    tts = TextToSpeech()
    
    example_text = 'Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности!'
    
    print(f"Converting to speech: {example_text}\n")
    tts.text_to_speech(example_text, play=True)
    print("Audio playback complete!")


if __name__ == "__main__":
    main()
