from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Reference Catalog Assets passed to the LLM context frame

# Expanded Reference Catalog Assets (23 Entries Total)
OFFER_CATALOG = [
    # --- DATA DOMINANT BOOSTERS & PACKS ---
    {
        "id": "DATA_BOOST_5G_MINI",
        "name": "5G DATA BOOSTER MINI",
        "type": "DATA",
        "price": 5.00,
        "description": "5GB ultra-speed 5G data allowance. 48-hour validity. Great for unexpected short-term data exhaustion.",
        "requires_roaming": False,
        "target_trigger": "Data depletion > 80% on a lower-spend account tier.",
    },
    {
        "id": "DATA_BOOST_5G",
        "name": "ULTRA 5G DATA BOOSTER",
        "type": "DATA",
        "price": 15.00,
        "description": "50GB high-speed 5G data allocation. 7-day validity.",
        "requires_roaming": False,
        "target_trigger": "High data depletion imminent on a mid-to-high spend account tier.",
    },
    {
        "id": "DATA_MAX_INFINITE",
        "name": "DATA MAX UNLIMITED TIER",
        "type": "DATA",
        "price": 85.00,
        "description": "True unlimited unthrottled 5G data for 30 days. Includes 20GB of mobile hotspot parsing.",
        "requires_roaming": False,
        "target_trigger": "Consistently extreme monthly data volume (>100GB) approaching contract rollover.",
    },
    {
        "id": "MIDNIGHT_OWL_DATA",
        "name": "NIGHT OWL UNLIMITED PASS",
        "type": "DATA",
        "price": 4.00,
        "description": "Unlimited unthrottled data usage everyday between 12:00 AM and 6:00 AM. Valid for 7 days.",
        "requires_roaming": False,
        "target_trigger": "High nocturnal telemetry data patterns or heavy off-peak usage flags.",
    },
    {
        "id": "WEEKEND_BINGE_PASS",
        "name": "WEEKEND STREAM BINGE PASS",
        "type": "DATA",
        "price": 8.00,
        "description": "Unlimited streaming data for YouTube, Netflix, and Twitch from Friday midnight to Sunday midnight.",
        "requires_roaming": False,
        "target_trigger": "Heavy weekend data traffic spikes observed in historical patterns.",
    },
    # --- INTERNATIONAL & ROAMING OPERATIONS ---
    {
        "id": "GLOBAL_ROAM_PASS",
        "name": "GLOBAL ROAM FREEDOM PASS",
        "type": "ROAMING",
        "price": 45.00,
        "description": "Unlimited roaming data and 200 minutes overseas voice capability in 120+ countries.",
        "requires_roaming": True,
        "target_trigger": "Subscriber operating out of home network zone with mid-to-high historical spend.",
    },
    {
        "id": "NEIGHBOR_ZONE_ROAM",
        "name": "BORDER ZONE LIGHT ROAM",
        "type": "ROAMING",
        "price": 12.00,
        "description": "2GB regional roaming data and 30 local minutes for cross-border neighboring territories. Valid for 3 days.",
        "requires_roaming": True,
        "target_trigger": "Frequent short-duration cross-border network handshakes detected.",
    },
    {
        "id": "INT_CALL_BUNDLE_APAC",
        "name": "APAC DIRECT VOICE LINK",
        "type": "ROAMING",
        "price": 18.00,
        "description": "500 dedicated minutes for outbound long-distance calls to Asia-Pacific international destinations.",
        "requires_roaming": False,
        "target_trigger": "High international domestic-outbound voice usage profiles.",
    },
    {
        "id": "EURO_TRAVEL_VOICE_DATA",
        "name": "EURO-PASSPORT TRAVEL MAX",
        "type": "ROAMING",
        "price": 60.00,
        "description": "25GB data allocation and unlimited voice roaming operations anywhere within the EU zone. Valid 14 days.",
        "requires_roaming": True,
        "target_trigger": "Active roaming connection verified inside EU country codes.",
    },
    # --- VOICE CENTRIC & VALUE SOLUTIONS ---
    {
        "id": "BUDGET_TALK_VOICE",
        "name": "INFINITE VOICE TALK PACK",
        "type": "VOICE",
        "price": 7.50,
        "description": "Unlimited domestic off-network minutes. 30-day validity.",
        "requires_roaming": False,
        "target_trigger": "Low voice balance or heavy legacy voice-centric telemetry history.",
    },
    {
        "id": "ESSENTIAL_PREPAID_TOPUP",
        "name": "PREPAID BASELINE SAFETY NET",
        "type": "VOICE",
        "price": 3.00,
        "description": "$5.00 worth of flexible emergency runtime talk-time and 500MB data fallback. No expiry.",
        "requires_roaming": False,
        "target_trigger": "Prepaid account balance drops below $0.50 threshold.",
    },
    {
        "id": "SENIOR_CONNECT_LIGHT",
        "name": "SILVER CONNECT LIFE PACK",
        "type": "VOICE",
        "price": 10.00,
        "description": "Unlimited on-network calls, 200 off-network minutes, and text alerts. 30-day validity.",
        "requires_roaming": False,
        "target_trigger": "Low-volume data consumption paired with highly consistent day-time voice patterns.",
    },
    {
        "id": "SMS_MARKET_POWER_PACK",
        "name": "METRO TEXT ESSENTIALS 5K",
        "type": "VOICE",
        "price": 4.50,
        "description": "5000 local and national text messages. Ideal for automated notifications or high text utility users.",
        "requires_roaming": False,
        "target_trigger": "Disproportionate volume of SMS interactions relative to data or voice baseline metrics.",
    },
    # --- ENTERPRISE, REMOTE WORK & HYBRID VARIATIONS ---
    {
        "id": "REMOTE_WORK_BOOST",
        "name": "WORKSPACE PRO PRODUCTIVITY ADD-ON",
        "type": "DATA",
        "price": 25.00,
        "description": "Priority routing paths for Zoom, Teams, and Slack traffic. Includes 30GB static tethering overhead.",
        "requires_roaming": False,
        "target_trigger": "Consistent business-hours data usage spikes indicating remote operations.",
    },
    {
        "id": "GIG_ECONOMY_LINK",
        "name": "GIG-DRIVE REALTIME NAVIGATION PACK",
        "type": "DATA",
        "price": 14.00,
        "description": "Zero-rated data metrics when processing Google Maps, Uber Driver, and delivery app routing matrix platforms.",
        "requires_roaming": False,
        "target_trigger": "Prolonged low-bandwidth telemetry active across geographic transport corridors.",
    },
    {
        "id": "ENTERPRISE_SECURE_TUNNEL",
        "name": "SECURE SHIELD BUSINESS TUNNEL",
        "type": "DATA",
        "price": 35.00,
        "description": "Enables zero-log network-level hardware VPN configurations and encryption wrapping for mobile endpoints.",
        "requires_roaming": False,
        "target_trigger": "Registered B2B profile flag exhibiting dynamic system administrative interaction metrics.",
    },
    # --- ENTERTAINMENT & VALUE-ADDED COMBO PIPELINES ---
    {
        "id": "STREAM_COMBO_PREMIUM",
        "name": "CYBER-ENTERTAINMENT VALUE CORE",
        "type": "DATA",
        "price": 29.00,
        "description": "Integrated 1-month platform passes for Apple Music, Spotify, and Disney+ alongside 20GB bonus pipeline allocation.",
        "requires_roaming": False,
        "target_trigger": "High-volume data parsing categorized inside entertainment server routes.",
    },
    {
        "id": "GAMING_LOW_LATENCY",
        "name": "FAST-TRACK FAST-PING GAMING BUFFER",
        "type": "DATA",
        "price": 20.00,
        "description": "Activates Quality of Service (QoS) optimizations on cellular towers to reduce jitter and latency for multiplayers.",
        "requires_roaming": False,
        "target_trigger": "Sustained high UDP data packet distribution segments indicating mobile gaming.",
    },
    {
        "id": "SOCIAL_MEDIA_PASS",
        "name": "INFINITE SOCIAL FREEDOM PASS",
        "type": "DATA",
        "price": 10.00,
        "description": "Unlimited usage of Instagram, TikTok, Facebook, and X. Does not consume core package quotas.",
        "requires_roaming": False,
        "target_trigger": "High data consumption mapped directly to major social CDN endpoints.",
    },
    # --- HYBRID REVENUE OPTIMIZATION TOP-UPS ---
    {
        "id": "FAMILY_POOL_TOPUP",
        "name": "FAMILY LINK MULTI-LINE SHARE 40G",
        "type": "DATA",
        "price": 40.00,
        "description": "40GB shared matrix data allocation injected directly into primary shared household account configurations.",
        "requires_roaming": False,
        "target_trigger": "Multi-line customer account profile demonstrating shared usage depletion.",
    },
    {
        "id": "CRITICAL_OVERAGE_SHIELD",
        "name": "AUTOMATED OVERAGE SHIELD 5GB",
        "type": "DATA",
        "price": 6.50,
        "description": "Emergency 5GB block that triggers automatically to intercept expensive pay-as-you-go rate configurations.",
        "requires_roaming": False,
        "target_trigger": "User is exactly at 100% data depletion and starting to incur baseline raw overage fees.",
    },
    {
        "id": "FIBER_BACKUP_LINK",
        "name": "HOME FIBER FALLBACK BACKUP 100G",
        "type": "DATA",
        "price": 50.00,
        "description": "100GB high-priority static data allocation meant to replace home broadband outages via active routing.",
        "requires_roaming": False,
        "target_trigger": "High tethering volumes combined with a sudden drop in detected local Wi-Fi profiles.",
    },
    {
        "id": "STUDENT_STUDY_BUNDLE",
        "name": "CAMPUS DIGITAL STUDENT PASS",
        "type": "DATA",
        "price": 11.00,
        "description": "Unlimited access to educational repositories, university domains, and e-learning portals plus 15GB generic data.",
        "requires_roaming": False,
        "target_trigger": "Subscriber age metric < 25 or telemetry matches campus cell tower IDs.",
    },
]


def evaluate_personalized_offer_via_llm(telemetry_payload):
    """
    Orchestrates the decision loop using Groq LLM inference,
    matching the execution strategy of your geospatial routing function.
    """
    # Instruct the LLM using an explicit, bounded system payload prompt
    prompt = f"""
    You are an expert Telecom Personalization and Revenue Optimization Engine.

    I have a JSON array containing an active subscriber's real-time telemetry profile, along with our current marketing offer catalog.

    **SUBSCRIBER METRICS:**
    {json.dumps(telemetry_payload, indent=4)}

    **OFFER CATALOG MATRIX:**
    {json.dumps(OFFER_CATALOG, indent=4)}

    **TASK:**
    Evaluate the user's immediate operational context constraints to find the single most optimal offer that increases revenue conversion potential while preventing irrelevant alignment (e.g., matching a high data depletion power user with a voice pack promo is an absolute failure mode).

    **DECISION MATRIX CRITERIA:**
    1. If `is_roaming` is true, heavily prioritize packs where `requires_roaming` is true.
    2. If `data_depletion_pct` is high (>70%), prioritize 'DATA' type offerings.
    3. Do not match high-value offers (price > $20) to users whose `monthly_spend` tier baseline is low (< $30), unless critical conditions (like roaming parameters) justify the expense.
    4. Compute a mathematical match 'score' from 1.0 to 10.0 for every option.

    **REQUIRED OUTPUT FORMAT (JSON ONLY):**
    {{
        "recommendations": [
            {{
                "offer_id": "CHOSEN_ID_1",
                "name": "OFFER NAME MATCHING CATALOG",
                "type": "CAT_TYPE",
                "price": 0.00,
                "description": "Catalog description text",
                "score": 9.2,
                "generated_pitch": "A highly contextual 1-sentence sales pitch typed in ALL-CAPS tailored specifically to their active telemetry variables."
            }}
        ]
    }}

    **CRITICAL SYSTEM INSTRUCTIONS:**
    1. NO MARKUP: Do not wrap the output in Markdown blocks (e.g., do not use ```json).
    2. RAW STRING ONLY: Your entire response must be a single, valid JSON object matching the schema. No conversational headers or code block tags.
    """

    # Initiate model using the exact strategy from your sample code
    # Adjust model name string to your preferred active Groq tag (e.g., 'llama3-70b-8192')
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    messages = [("system", prompt)]

    response = llm.invoke(messages)

    try:
        # Strip trailing/leading accidental strings if any, then load
        clean_content = response.content.strip()
        parsed_json = json.loads(clean_content)
        return parsed_json
    except Exception as e:
        # Emergency local fallback engine loop to prevent UI failure in case of parsing exceptions
        print(
            f"Parsing Exception triggered: {e}. Executing emergency local matrix fallback."
        )
        return {
            "recommendations": [
                {
                    "offer_id": "DATA_BOOST_5G",
                    "name": "ULTRA 5G DATA BOOSTER [FALLBACK]",
                    "type": "DATA",
                    "price": 15.00,
                    "description": "50GB high-speed 5G data allocation. 7-day validity.",
                    "score": 5.0,
                    "generated_pitch": "SYSTEM OPERATING IN BACKUP MODE. TELEMETRY CRITICAL.",
                }
            ]
        }


@app.route("/api/recommend", methods=["POST"])
def recommend_offers():
    input_data = request.json or {}

    # Bundle user variables
    telemetry_payload = {
        "user_id": input_data.get("user_id", "SUB-8849"),
        "data_usage_gb": float(input_data.get("data_usage_gb", 45.0)),
        "data_depletion_pct": float(input_data.get("data_depletion_pct", 82.0)),
        "is_roaming": bool(input_data.get("is_roaming", False)),
        "monthly_spend": float(input_data.get("monthly_spend", 60.0)),
        "historical_preference": input_data.get("historical_preference", "DATA"),
    }

    # Pass logic execution to LLM router agent
    llm_decision = evaluate_personalized_offer_via_llm(telemetry_payload)

    # Inject runtime headers matching frontend structural requirements
    response_payload = {
        "timestamp": int(time.time()),
        "target_subscriber": telemetry_payload["user_id"],
        "telemetry_processed": {
            "roaming_status": "ACTIVE"
            if telemetry_payload["is_roaming"]
            else "DOMESTIC",
            "depletion": f"{telemetry_payload['data_depletion_pct']}%",
        },
        "recommendations": llm_decision.get("recommendations", []),
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    # Default port matching your React frontend configurations
    app.run(port=5000, debug=True)
