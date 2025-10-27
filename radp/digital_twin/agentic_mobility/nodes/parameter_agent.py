"""LangGraph node for parameter generation."""
from typing import Dict

from radp.digital_twin.agentic_mobility.chains.parameter_chain import ParameterChain
from radp.digital_twin.agentic_mobility.models.query_intent import QueryIntent
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def node(state: MobilityGenerationState) -> Dict:
    """Parameter agent node - generates mobility parameters using LLM.

    Args:
        state: Current graph state

    Returns:
        Dict with only gen_params key updated
    """
    query_intent_dict = state["query_intent"]

    if not query_intent_dict:
        raise ValueError("query_intent must be populated before parameter generation")

    # Convert dict back to QueryIntent object
    query_intent = QueryIntent(**query_intent_dict)

    parameter_chain = ParameterChain()

    # Generate parameters
    gen_params = parameter_chain.generate(query_intent)

    # Return only the keys we're updating
    return {"gen_params": gen_params.dict()}
