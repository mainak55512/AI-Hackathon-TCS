from langchain_groq import ChatGroq
from flask import Flask
import pandas as pd
import json

# import requests

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def get_safest_path():
    safest_path = get_path()
    return " ------> ".join(map(surround, safest_path))


def surround(e):
    return f"[{e}]"


def get_details(df, segment_id, field):
    return df[df["segment_id"] == segment_id][field].item()


def path_details_builder(df, path_list):
    path_details = {}
    for elem in path_list:
        path_details[elem] = {
            "lat": get_details(df, elem, "latitude"),
            "long": get_details(df, elem, "longitude"),
            "lux": get_details(df, elem, "lighting_lux"),
            "footfall": get_details(df, elem, "footfall_avg"),
            "incident_density": get_details(df, elem, "incident_density"),
            "cctv_coverage": get_details(df, elem, "cctv_proxy"),
        }

    return path_details


def get_path():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    df = pd.read_excel("data.xlsx")

    # Creating list of nodes as path
    # as I don't know what the existing paths are
    path1 = ["KOL_001", "KOL_004", "KGP_010"]
    path2 = ["KOL_001", "KOL_003", "KOL_007", "KGP_009", "KGP_010"]
    path3 = ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_010"]
    path4 = ["KOL_001", "KOL_003", "KOL_005", "KGP_008", "KGP_010"]

    path_1_details = path_details_builder(df, path1)
    path_2_details = path_details_builder(df, path2)
    path_3_details = path_details_builder(df, path3)
    path_4_details = path_details_builder(df, path4)

    all_path = [path_1_details, path_2_details, path_3_details, path_4_details]

    # print(json.dumps(all_path, indent=4))

    prompt = f"""
    You are a geospatial risk analysis expert.

    I have a JSON payload containing an array of potential paths.
    Each path consists of nodes with the following attributes:
    'lat' (Latitude), 'long' (Longitude), 'lux' (Lighting level), 'footfall' (Crowd density), 
    'incident_density' (Historical crime/incident frequency) 
    and 'cctv_coverage' (0.0 to 1.0).

    **Task:**
    Apply Dijkstra's algorithm to find the **safest** path from the starting node to the destination.
    DO NOT SKIP any NODE/Segment, traverse all the nodes of a path when doing calculation.
    Output should also contain the complete path

    **Decision Logic:**
    1. Prioritize **Safety Score** over physical distance.
    2. Calculate safety using: Higher 'lux' and 'cctv_coverage' are positive; higher 'incident_density' and 'footfall' is strongly negative.
    3. The cost function for Dijkstra should heavily penalize segments with low lighting (< 20 lux) or high incident density (> 0.15).

    **Input Payload:**
    {json.dumps(all_path, indent=4)}

    **Required Output Format (JSON only):**
    {{
        "safest_path": ["node_id_1", "node_id_2", ...],
        "reason": "A detailed explanation of why this path was chosen based on the safety parameters compared to the other options"
    }}

    **CRITICAL INSTRUCTIONS:**
    1. NO MARKUP: Do not wrap the output in Markdown code blocks (e.g., do not use ```json).
    2. RAW STRING: Your entire response must be a single, valid JSON object and nothing else.
    """

    # print(prompt)

    # payload = {
    #     "messages": [{"role": "user", "content": prompt}],
    #     "model": "openai/gpt-oss-120b",
    # }

    # headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    # response = requests.post(
    #     "https://api.groq.com/openai/v1/chat/completions",
    #     data=json.dumps(payload),
    #     headers=headers,
    # )

    # resp_obj = json.loads(response.json()["choices"][0]["message"]["content"])
    # return resp_obj["safest_path"]

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=None)
    messages = [("system", prompt)]

    response = llm.invoke(messages)
    return json.loads(response.content)["safest_path"]


if __name__ == "__main__":
    app.run(debug=True)
