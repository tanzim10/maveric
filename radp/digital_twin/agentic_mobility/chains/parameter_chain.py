"""LLM chain for parameter generation."""
from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import QueryIntent
from radp.digital_twin.agentic_mobility.prompts.parameter_prompts import PARAMETER_SYSTEM_PROMPT, get_parameter_prompt
from radp.digital_twin.agentic_mobility.utils.llm_client import LLMClient


class ParameterChain:
    """Chain for generating mobility parameters using LLM-based direct generation.

    NO TOOLS - uses single LLM call to generate all parameters.
    """

    def __init__(self):
        """Initialize parameter chain."""
        self.llm_client = LLMClient()

    def generate(self, query_intent: QueryIntent) -> GenParams:
        """Generate mobility parameters from QueryIntent.

        Args:
            query_intent: Parsed query intent

        Returns:
            GenParams object with generated parameters

        Raises:
            Exception: If generation fails after retries
        """
        # Convert ue_distribution from MobilityClass keys to string keys if provided
        ue_dist_dict = None
        if query_intent.ue_distribution:
            ue_dist_dict = {k.value: v for k, v in query_intent.ue_distribution.items()}

        prompt = get_parameter_prompt(
            scenario_type=query_intent.scenario_type.value,
            num_ues=query_intent.num_ues,
            num_ticks=query_intent.num_ticks,
            raw_query=query_intent.raw_query,
            ue_distribution=ue_dist_dict,
        )

        gen_params = self.llm_client.generate_structured(
            prompt=prompt, output_model=GenParams, system_message=PARAMETER_SYSTEM_PROMPT
        )

        return gen_params
