"""
src/llm_tools.py
Generalized LLM tooling for handling model agnostic conversations and tooling.
"""
import json
from typing import Any
from typing import Dict
from typing import List

import anthropic
from google import genai
from openai import OpenAI


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
        self.usage_logger = None
        self.usage_context: Dict[str, Any] = {
            "step_type": None, "iteration": None, "attempt": None,
        }

    def chat(self, user_msg: str, model: str) -> str:
        if not user_msg or not model:
            return ""
        self.__user(user_msg)
        self.last_usage = None

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

    def set_usage_logger(self, logger) -> None:
        """Attach an LLMUsageLogger. Pass None to disable."""
        self.usage_logger = logger

    def set_usage_context(self, *, step_type=None, iteration=None, attempt=None) -> None:
        self.usage_context = {
            "step_type": step_type,
            "iteration": iteration,
            "attempt": attempt,
        }

    def _record_gemini_usage(self, model: str, response) -> None:
        try:
            u = getattr(response, "usage_metadata", None)
            if u is None:
                return
            self._record_usage("google", model, {
                "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(u, "candidates_token_count", 0) or 0,
                "reasoning_tokens": getattr(u, "thoughts_token_count", 0) or 0,
            })
        except Exception:
            pass

    def _record_usage(self, provider: str, model: str, usage: Dict[str, int]) -> None:
        if self.usage_logger is None:
            return
        try:
            self.usage_logger.log(
                step_type=self.usage_context.get("step_type"),
                iteration=self.usage_context.get("iteration"),
                attempt=self.usage_context.get("attempt"),
                provider=provider,
                model=model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
            )
        except Exception:
            pass

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

            try:
                u = getattr(message, "usage", None)
                if u is not None:
                    self._record_usage("anthropic", model, {
                        "input_tokens": getattr(u, "input_tokens", 0) or 0,
                        "output_tokens": getattr(u, "output_tokens", 0) or 0,
                        "reasoning_tokens": 0,
                    })
            except Exception:
                pass

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
                self._record_gemini_usage(model, response)
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

            self._record_gemini_usage(model, response)
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

        try:
            self._openai_client = OpenAI()
            messages = self.__to_openai_messages()

            # Make the API call
            response = self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=4096
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = getattr(details, "reasoning_tokens", 0) if details else 0
                self.last_usage = {
                    "provider": "openai",
                    "model": model,
                    "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "reasoning_tokens": int(reasoning or 0),
                }
            try:
                u = getattr(response, "usage", None)
                if u is not None:
                    reasoning = 0
                    details = getattr(u, "completion_tokens_details", None)
                    if details is not None:
                        reasoning = getattr(details, "reasoning_tokens", 0) or 0
                    self._record_usage("openai", model, {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": reasoning,
                    })
            except Exception:
                pass

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
