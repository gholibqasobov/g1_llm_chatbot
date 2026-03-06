#!/usr/bin/env python3
"""
LLM Chat module using OpenAI-compatible API (e.g., Ollama).

Requirements:
- ollama serve
- ollama pull qwen2.5:7b-instruct-q4_K_M
"""

from openai import OpenAI
import os


class LLMChatbot:
    """Chat interface using OpenAI-compatible LLM service."""

    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        system_prompt=None,
        temperature=0.4,
        max_tokens=300,
    ):
        """
        Initialize the LLM Chatbot.

        Args:
            base_url: LLM service base URL (default: localhost:11434)
            api_key: API key for the service (default: "ollama")
            model: Model name (default: qwen3:4b-instruct-2507-q4_K_M)
            system_prompt: System prompt for the AI personality
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.model = model or os.getenv("LLM_CHOICE", "qwen3:4b-instruct-2507-q4_K_M")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = """
Ты — Жорик, дружелюбный робот от иксбот. Твой стиль: легкий, позитивный и человечный.
Правила:

    Пиши кратко (1-2 предложения).

    Если не знаешь имени — спроси.

    Интересуйся делами пользователя и задавай встречные вопросы.

    Не читай лекций, если не просят; предлагай идеи и шути.

    Будь уверенным, но простым в общении. Не создавай смайлики, будь более формальным и вежливым
"""

        self.system_prompt = system_prompt.strip()

        # Create OpenAI client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Conversation memory
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_response(self, user_message):
        """
        Generate a response to the user message.

        Args:
            user_message: The user's input text

        Returns:
            str: The assistant's response
        """
        # Add user message to memory
        self.messages.append({"role": "user", "content": user_message})

        # Stream response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        assistant_reply = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                assistant_reply += token

        # Save assistant response in memory
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def clear_history(self):
        """Clear conversation history (keeping system prompt)."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_history(self):
        """Get current conversation history."""
        return self.messages.copy()


def main():
    """Interactive chat mode."""
    print("=" * 60)
    print("KaspiBot Assistant")
    print("Type /exit to quit")
    print("=" * 60)

    chatbot = LLMChatbot()
    print(f"Model: {chatbot.model}")
    print("Make sure Ollama is running: ollama serve\n")

    while True:
        try:
            user_input = input("\nYou: ")

            if user_input.strip().lower() in ["/exit", "exit", "quit"]:
                print("Goodbye! До свидания!")
                break

            print("Assistant: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\nInterrupted. До свидания!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            print("Make sure Ollama is running and the model is available.")


if __name__ == "__main__":
    main()
