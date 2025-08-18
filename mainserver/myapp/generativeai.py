# v1.0.1

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Access the variables
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("LLM_MODEL")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# def generate_prompt(data):
#     prompt = (
#         f"You are a biology expert helping analyze plant cell ultrastructure segmentation data. "
#         f"Here is the analysis:\n\n"
#         f"- Etioplast Area: {data['Etioplast']['area_um2']} µm² ({data['Etioplast']['count']} regions)\n"
#         f"- PLB Area: {data['PLB']['area_um2']} µm² ({data['PLB']['count']} regions)\n"
#         f"- Prothylakoid Total Length: {data['Prothylakoid']['total_length_um']} µm "
#         f"({data['Prothylakoid']['count']} regions)\n"
#         f"- Plastoglobule Avg. Diameter: {data['Plastoglobule']['avg_diameter_um']} µm "
#         f"({data['Plastoglobule']['count']} regions)\n\n"
#         f"Please summarize these findings in simple biological terms and explain what they might suggest about the sample's plastid structure."
#     )
#     return prompt

# generativeai.py

def generate_prompt(analysis):
    eti = analysis.get("Etioplast", {})
    plb = analysis.get("PLB", {})
    pro = analysis.get("Prothylakoid", {})
    pg  = analysis.get("Plastoglobule", {})
    sg  = analysis.get("StarchGain", {})  # note: your key is "StarchGain"

    eti_area = float(eti.get("total_area_um2", 0.0))
    eti_cnt  = int(eti.get("count", 0))

    plb_area = float(plb.get("total_area_um2", 0.0))
    plb_cnt  = int(plb.get("count", 0))

    pro_len  = float(pro.get("total_length_um", 0.0))
    pro_cnt  = int(pro.get("count", 0))

    pg_diam  = float(pg.get("diameter_um", 0.0))   # your field name
    pg_cnt   = int(pg.get("count", 0))

    sg_area  = float(sg.get("total_area_um2", 0.0))
    sg_cnt   = int(sg.get("count", 0))

    prompt = (
        "You are a biology expert helping analyze plant cell ultrastructure segmentation data.\n\n"
        f"- Etioplast Area (total): {eti_area:.3f} µm² ({eti_cnt} regions)\n"
        f"- PLB Area (total): {plb_area:.3f} µm² ({plb_cnt} regions)\n"
        f"- Prothylakoid Total Length: {pro_len:.3f} µm ({pro_cnt} regions)\n"
        f"- Plastoglobule Avg. Diameter: {pg_diam:.3f} µm ({pg_cnt} regions)\n"
        f"- Starch (total area): {sg_area:.3f} µm² ({sg_cnt} regions)\n\n"
        "Please summarize these findings in simple biological terms and explain what they might "
        "suggest about the sample's plastid structure."
    )
    return prompt


def get_generative_response(data):
    prompt = generate_prompt(data)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful biology assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM API error {response.status_code}: {response.text}")
