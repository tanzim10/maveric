"""LangGraph node for suggestion generation."""
from radp.digital_twin.agentic_mobility.chains.suggestion_chain import SuggestionChain
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def node(state: MobilityGenerationState) -> MobilityGenerationState:
    """Suggestion node - generates suggestions from validation failures.

    Args:
        state: Current graph state

    Returns:
        Updated state with current_query augmented and retry_count incremented
    """
    validation_result = state["validation_result"]
    user_query = state["user_query"]
    retry_count = state["retry_count"]

    if not validation_result or validation_result["status"] == "success":
        # No suggestions needed
        return state

    suggestion_chain = SuggestionChain()

    # Generate augmented query
    augmented_query = suggestion_chain.generate_suggestion(
        original_query=user_query,
        validation_errors=validation_result.get("validation_errors", []),
        failure_reasons=validation_result.get("failure_reasons", {}),
        retry_count=retry_count,
    )

    # Update state
    state["current_query"] = augmented_query
    state["retry_count"] = retry_count + 1

    return state
