from router.semantic_router import SemanticRouter
from router.cascade_router import CascadeRouter

# Configuration for semantic intent clusters
SEMANTIC_ROUTES = {
    "simple_retrieval": {
        "samples": [
            "What is the capital of France?",
            "Who won the 1998 World Cup?",
            "Define photosynthesis briefly."
        ],
        "target_model": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_1k": 0.00015
    },
    "coding_architecture": {
        "samples": [
            "Write an asynchronous event loop in Rust",
            "Refactor this SQL query with subqueries into CTEs",
            "Implement a Kubernetes controller with client-go"
        ],
        "target_model": "claude-3-5-sonnet",
        "provider": "anthropic",
        "cost_per_1k": 0.003
    }
}

def main():
    print("--- 1. Semantic Router Test ---")
    semantic = SemanticRouter(routes=SEMANTIC_ROUTES)
    
    test_queries = [
        "Where is the headquarters of Interpol?",
        "Write a python script to parse binary heap memory dumps.",
        "Compose an existential poem about space."
    ]

    for q in test_queries:
        decision = semantic.route(q)
        print(f"Query: '{q}'\n-> Route: {decision.model_name} ({decision.provider}) | Reason: {decision.reason} | Conf: {decision.confidence:.2f}\n")

    print("--- 2. Cascade Router Test ---")
    # Validator: Ensures response does not contain uncertainty markers
    validator = lambda resp: "I am not sure" not in resp and len(resp.strip()) > 20
    cascade = CascadeRouter(evaluator_fn=validator)

    initial_route = cascade.route("Explain quantum supremacy")
    print(f"Initial Dispatch: {initial_route.model_name}")

    mock_weak_response = "I am not sure how to answer this accurately."
    escalation = cascade.evaluate_or_escalate(mock_weak_response)
    if escalation:
        print(f"Escalated to: {escalation.model_name} (Reason: {escalation.reason})")

if __name__ == "__main__":
    main()
