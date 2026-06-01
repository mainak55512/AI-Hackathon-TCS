import os
import json
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# Define the shared state dictionary passed across agents in the graph
class AgentState(TypedDict):
    subscriber_id: str
    raw_telemetry: Dict[str, Any]
    crm_billing_profile: Dict[str, Any]
    eligible_offers: List[Dict[str, Any]]
    selected_offer: Dict[str, Any]
    final_output_payload: Dict[str, Any]


class TelecomMultiAgentSystem:
    def __init__(self):
        # High-performance, low-latency inference engine for copy generation
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

        # In-memory mock databases mimicking CRM, Billing, and Catalog Microservices
        self.mock_catalog = [
            {
                "id": "DATA_BOOST_5G",
                "name": "ULTRA 5G DATA BOOSTER",
                "type": "DATA",
                "price": 15.00,
                "margin": 11.50,
            },
            {
                "id": "GLOBAL_ROAM_PASS",
                "name": "GLOBAL ROAM FREEDOM PASS",
                "type": "ROAMING",
                "price": 45.00,
                "margin": 32.00,
            },
            {
                "id": "BUDGET_TALK_VOICE",
                "name": "INFINITE VOICE TALK PACK",
                "type": "VOICE",
                "price": 7.50,
                "margin": 5.00,
            },
            {
                "id": "CRITICAL_OVERAGE_SHIELD",
                "name": "AUTOMATED OVERAGE SHIELD 5GB",
                "type": "DATA",
                "price": 6.50,
                "margin": 4.80,
            },
        ]

    def telemetry_broker_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 1: Hydrates raw data streams by querying CRM & Billing microservices."""
        sub_id = state["subscriber_id"]

        # Simulating automated backend API requests
        mock_crm_response = {
            "tier": "PLATINUM"
            if state["raw_telemetry"].get("monthly_spend", 0) > 50
            else "STANDARD",
            "contract_type": "PREPAID",
            "historical_preference": state["raw_telemetry"].get(
                "historical_preference", "DATA"
            ),
        }
        return {"crm_billing_profile": mock_crm_response}

    def predictive_scoring_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 2: Runs predictive heuristics to calculate acceptance probability."""
        telemetry = state["raw_telemetry"]
        pref = state["crm_billing_profile"].get("historical_preference", "DATA")
        scored_list = []

        for offer in self.mock_catalog:
            p_accept = 0.05  # Lower baseline floor to let context shine
            relevancy_multiplier = 0.1  # Heavy penalty factor if no conditions match

            # 1. Evaluate Data Overage Urgency
            if offer["type"] == "DATA":
                depletion = telemetry.get("data_depletion_pct", 0)
                if depletion > 90 and offer["id"] == "CRITICAL_OVERAGE_SHIELD":
                    p_accept += 0.85
                    relevancy_multiplier = 2.0
                elif depletion > 70 and offer["id"] == "DATA_BOOST_5G":
                    p_accept += 0.70
                    relevancy_multiplier = 1.5

            # 2. Evaluate Roaming Context (Should instantly override everything else when true)
            elif offer["type"] == "ROAMING" and telemetry.get("is_roaming", False):
                p_accept += 0.85
                relevancy_multiplier = 3.0

            # 3. Evaluate Voice Centricity
            elif offer["type"] == "VOICE" and pref == "VOICE":
                p_accept += 0.65
                relevancy_multiplier = 1.8

            # Fallback alignment check for historical choice
            if offer["type"] == pref:
                p_accept += 0.10
                relevancy_multiplier = max(relevancy_multiplier, 1.2)

            scored_list.append(
                {
                    **offer,
                    "p_accept": min(p_accept, 0.99),
                    "relevancy_multiplier": relevancy_multiplier,
                }
            )

        return {"eligible_offers": scored_list}

    def financial_roi_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 3: Optimizes for business margin balanced against explicit context relevancy multipliers."""
        eligible = state["eligible_offers"]
        spend_limit = state["raw_telemetry"].get("monthly_spend", 40.0)

        best_offer = None
        max_expected_value = -999.0

        for offer in eligible:
            # Budget Over-stretch Protection Guardrail
            if offer["price"] > (spend_limit * 1.1):
                continue

            # NEW BALANCED FORMULA: Factor in the contextual relevancy weight
            # Expected Value = Probability * Margin * Contextual Urgency Multiplier
            expected_val = (
                offer["p_accept"]
                * offer["margin"]
                * offer.get("relevancy_multiplier", 1.0)
            )

            if expected_val > max_expected_value:
                max_expected_value = expected_val
                best_offer = offer

        return {"selected_offer": best_offer}

    def creative_pitch_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 4: Generates tailored, high-converting marketing copy via LLM."""
        offer = state["selected_offer"]
        telemetry = state["raw_telemetry"]

        prompt = f"""
        You are an advanced conversion copywriting microservice.
        Generate a single-sentence marketing notification popup for a telecom application dashboard.
        
        Target Offer Selected: {offer["name"]}
        Active Context: User has {telemetry["data_depletion_pct"]}% data exhaustion, Roaming status: {telemetry["is_roaming"]}.
        
        CRITICAL RULES:
        1. Write the message in ALL-CAPS to fit a cyber-terminal theme.
        2. Keep it under 25 words. Be punchy, direct, and mention the specific situational trigger.
        3. Do not include quotes or metadata. Return the raw copy string only.
        """

        response = self.llm.invoke([("system", prompt)])

        final_payload = {
            "offer_id": offer["id"],
            "name": offer["name"],
            "price": offer["price"],
            "p_accept_metric": f"{round(offer['p_accept'] * 100, 1)}%",
            "generated_pitch": response.content.strip(),
        }
        return {"final_output_payload": final_payload}

    def execute_pipeline(
        self, sub_id: str, telemetry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinates sequential execution paths across agents mimicking state graph execution loops."""
        state = AgentState(
            subscriber_id=sub_id,
            raw_telemetry=telemetry,
            crm_billing_profile={},
            eligible_offers=[],
            selected_offer={},
            final_output_payload={},
        )

        # Step 1: Ingest and Hydrate Context
        state.update(self.telemetry_broker_node(state))
        # Step 2: Score Acceptance Probabilities
        state.update(self.predictive_scoring_node(state))
        # Step 3: Run Financial Optimization Filters
        state.update(self.financial_roi_node(state))
        # Step 4: Generate Marketing Copy via Groq Engine
        state.update(self.creative_pitch_node(state))

        return state["final_output_payload"]
