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
        Dict with gen_params and updated query_intent
    """
    query_intent_dict = state["query_intent"]

    if not query_intent_dict:
        raise ValueError("query_intent must be populated before parameter generation")

    # Convert dict back to QueryIntent object
    query_intent = QueryIntent(**query_intent_dict)

    parameter_chain = ParameterChain()

    # Generate parameters and get updated query_intent with distribution source
    gen_params, updated_query_intent = parameter_chain.generate(query_intent)

    # Return updated gen_params AND updated query_intent (with distribution source)
    return {"gen_params": gen_params.dict(), "query_intent": updated_query_intent.dict()}
