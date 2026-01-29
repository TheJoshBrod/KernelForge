"""
src/llm_tools.py
Generalized LLM tooling for handling model agnostic conversations and tooling.
"""
import json
import os
from typing import Any
from typing import Dict
from typing import List

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from google import genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class GenModel:
    """
    Provider-agnostic chat history.

    - sys_prompt is stored separately
    - history contains only conversational turns w/ tool calls
    - tools are possible tools model can call
    """

    def __init__(self, sys_prompt: str):
        self.sys_prompt = sys_prompt
        self.history: List[Dict[str, Any]] = []
        self.tools: Dict[str, callable] = {}

    def chat(self, user_msg: str, model: str) -> str:
        if not user_msg or not model:
            return ""
        self.__user(user_msg)

        response = ""

        if "claude" in model:
            response = self.__claude(model)
        elif "gemini" in model:
            response = self.__gemini(model)
        elif "gpt" in model:
            response = self.__chatgpt(model)
        else:
            self.history.pop()
            return "Unsupported llm model/provider"

        self.__assistant(response)
        return response

    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = sys_prompt

    def set_tools(self, tools: Dict[str, callable]):
        self.tools = tools

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.history, **kwargs)

    def __repr__(self) -> str:
        return f"ChatHistory(turns={len(self.history)})"

    # Helper functions to interface with different LLM providers

    def __claude(self, model: str) -> str:
        """Call Anthropics's Claude API

        Args:
            model (str): Claude model name 

        Returns:
            str: LLM response
        """

        if anthropic is None:
            return "Error calling Claude API: anthropic package is not installed"

        payload = self.__to_anthropic_payload()

        try:
            # Make the API call
            self._anthropic_client = anthropic.Anthropic()
            message = self._anthropic_client.messages.create(
                model=model,
                max_tokens=8196,
                system=payload["system"],
                messages=payload["messages"]
            )

            # Extract text from response
            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text

            return response_text

        except Exception as e:
            return f"Error calling Claude API: {str(e)}"

    def __gemini(self, model: str) -> str:
        """Call Google's Gemini API

        Args:
            model (str): Gemini model name 

        Returns:
            str: LLM response
        """

        try:
            self._genai_client = genai.Client()

            # For first message, just generate content
            if len(self.history) == 1:
                response = self._genai_client.models.generate_content(
                    model=model,
                    contents=self.history[0]["content"],
                    config={
                        "system_instruction": self.sys_prompt
                    }
                )
                return response.text

            # For multi-turn conversations, use chat API
            # Convert history to Gemini format
            chat_history = []
            for msg in self.history[:-1]:  # Exclude the last user message
                role = "model" if msg["role"] == "assistant" else "user"
                chat_history.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            # Create chat with history
            chat = self._genai_client.chats.create(
                model=model,
                config={
                    "system_instruction": self.sys_prompt
                },
                history=chat_history
            )

            # Send the latest user message
            latest_user_msg = self.history[-1]["content"]
            response = chat.send_message(latest_user_msg)

            return response.text

        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

    def __chatgpt(self, model: str) -> str:
        """Call OpenAI's ChatGPT API

        Args:
            model (str): ChatGPT model name 

        Returns:
            str: LLM response
        """

        if OpenAI is None:
            return "Error calling OpenAI API: openai package is not installed"

        try:
            self._openai_client = OpenAI()
            if not model:
                model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
            messages = self.__to_openai_messages()

            # Make the API call
            use_responses_env = os.environ.get("OPENAI_USE_RESPONSES", "")
            if use_responses_env:
                use_responses = use_responses_env.lower() in {"1", "true", "yes"}
            else:
                use_responses = model.startswith("gpt-5")
            max_output = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
            max_tokens = os.environ.get("OPENAI_MAX_TOKENS")

            if use_responses:
                params = {
                    "model": model,
                    "input": messages,
                }
                if max_output:
                    params["max_output_tokens"] = int(max_output)
                response = self._openai_client.responses.create(**params)
                return response.output_text

            params = {
                "model": model,
                "messages": messages,
            }
            if max_output:
                params["max_output_tokens"] = int(max_output)
            elif max_tokens:
                params["max_tokens"] = int(max_tokens)

            try:
                response = self._openai_client.chat.completions.create(**params)
            except TypeError:
                if "max_output_tokens" in params:
                    params.pop("max_output_tokens", None)
                    params["max_tokens"] = int(max_output)
                    response = self._openai_client.chat.completions.create(**params)
                else:
                    raise

            return response.choices[0].message.content

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    # Helper functions to add different types of messages to chat history

    def __user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def __assistant(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def __tool(self, name: str, content: str) -> None:
        self.history.append({
            "role": "tool",
            "name": name,
            "content": content
        })

    # Helper functions to extract chat history format for generator

    def __to_openai_messages(self) -> List[Dict[str, Any]]:
        return (
            [{"role": "system", "content": self.sys_prompt}]
            + self.history
        )

    def __to_anthropic_payload(self) -> Dict[str, Any]:
        return {
            "system": self.sys_prompt,
            "messages": self.history
        }
