from flask import Flask, request, jsonify
from flask_cors import CORS
from agents_engine import TelecomMultiAgentSystem
import time

app = Flask(__name__)
CORS(app)

# Initialize our multi-agent core orchestration subsystem
agent_orchestrator = TelecomMultiAgentSystem()


@app.route("/api/recommend", methods=["POST"])
def process_stream_ingestion():
    start_time = time.time()
    input_payload = request.json or {}

    user_id = input_payload.get("user_id", "SUB-X7")
    telemetry = {
        "data_usage_gb": float(input_payload.get("data_usage_gb", 50)),
        "data_depletion_pct": float(input_payload.get("data_depletion_pct", 80)),
        "is_roaming": bool(input_payload.get("is_roaming", False)),
        "monthly_spend": float(input_payload.get("monthly_spend", 60)),
        "historical_preference": input_payload.get("historical_preference", "DATA"),
    }

    try:
        # Pass payload routing to multi-agent state machine execution
        computed_result = agent_orchestrator.execute_pipeline(user_id, telemetry)

        response_payload = {
            "status": "SUCCESS",
            "timestamp": int(time.time()),
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "target_subscriber": user_id,
            "telemetry_processed": {
                "roaming_status": "ACTIVE" if telemetry["is_roaming"] else "DOMESTIC",
                "depletion": f"{telemetry['data_depletion_pct']}%",
            },
            # Encapsulate generated results to cleanly hydrate the frontend
            "recommendations": [computed_result],
        }
        return jsonify(response_payload), 200

    except Exception as err:
        return jsonify(
            {
                "status": "CRITICAL_ERROR",
                "message": f"Pipeline tracking interrupted internally: {str(err)}",
            }
        ), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
