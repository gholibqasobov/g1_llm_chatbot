
from src.voice.speech_to_text import SpeechToText
from src.voice.text_to_speech import TextToSpeech
from src.chatbot.llm_chatbot import LLMChatbot


def main():

    print("=" * 60)
    print("🎤 Voice Chatbot Started")
    print("Press Ctrl+C to exit")
    print("=" * 60)

    stt = SpeechToText()
    tts = TextToSpeech()
    llm = LLMChatbot()

    tts.text_to_speech("Здравствуйте! меня зовут Жорик, как вас зовут?", play=True)

    while True:

        try:

            print("\nListening...")

            user_text = stt.listen()

            # ---- skip empty input ----
            if not user_text:
                print("No speech detected.")
                continue

            print("User:", user_text)

            response = llm.generate_response(user_text)

            print("Bot:", response)

            tts.text_to_speech(response, play=True)

        except KeyboardInterrupt:
            print("\nStopping assistant.")
            break

        except Exception as e:
            print("Runtime error:", e)

if __name__ == "__main__":
    main()
