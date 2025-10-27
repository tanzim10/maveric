"""Groq API client wrapper."""
from typing import Optional, Type, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from radp.digital_twin.agentic_mobility.config import Config

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wrapper for LLM API calls with structured output support."""

    def __init__(self):
        """Initialize LLM client with Groq configuration."""
        Config.validate()
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=0.0,  # Deterministic outputs for structured generation
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_structured(self, prompt: str, output_model: Type[T], system_message: Optional[str] = None) -> T:
        """Generate structured output using Pydantic model.

        Args:
            prompt: User prompt
            output_model: Pydantic model class for output
            system_message: Optional system message

        Returns:
            Instance of output_model

        Raises:
            Exception: If LLM call fails after retries
        """
        parser = PydanticOutputParser(pydantic_object=output_model)

        messages = []
        if system_message:
            messages.append(("system", system_message))

        # Add format instructions to prompt
        format_instructions = parser.get_format_instructions()
        full_prompt = f"{prompt}\n\n{format_instructions}"
        messages.append(("human", full_prompt))

        response = self.llm.invoke(messages)
        return parser.parse(response.content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text output.

        Args:
            prompt: User prompt
            system_message: Optional system message

        Returns:
            Generated text

        Raises:
            Exception: If LLM call fails after retries
        """
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))

        response = self.llm.invoke(messages)
        return response.content
