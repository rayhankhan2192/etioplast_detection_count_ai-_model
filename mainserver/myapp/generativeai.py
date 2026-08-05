# # v1.0.1

# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# # Access the variables
# API_KEY = os.getenv("API_KEY")
# API_URL = os.getenv("API_URL")
# MODEL = os.getenv("LLM_MODEL")

# HEADERS = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# # def generate_prompt(data):
# #     prompt = (
# #         f"You are a biology expert helping analyze plant cell ultrastructure segmentation data. "
# #         f"Here is the analysis:\n\n"
# #         f"- Etioplast Area: {data['Etioplast']['area_um2']} µm² ({data['Etioplast']['count']} regions)\n"
# #         f"- PLB Area: {data['PLB']['area_um2']} µm² ({data['PLB']['count']} regions)\n"
# #         f"- Prothylakoid Total Length: {data['Prothylakoid']['total_length_um']} µm "
# #         f"({data['Prothylakoid']['count']} regions)\n"
# #         f"- Plastoglobule Avg. Diameter: {data['Plastoglobule']['avg_diameter_um']} µm "
# #         f"({data['Plastoglobule']['count']} regions)\n\n"
# #         f"Please summarize these findings in simple biological terms and explain what they might suggest about the sample's plastid structure."
# #     )
# #     return prompt

# # generativeai.py

# def generate_prompt(analysis):
#     eti = analysis.get("Etioplast", {})
#     plb = analysis.get("PLB", {})
#     pro = analysis.get("Prothylakoid", {})
#     pg  = analysis.get("Plastoglobule", {})
#     sg  = analysis.get("StarchGain", {})  

#     eti_area = float(eti.get("total_area_um2", 0.0))
#     eti_cnt  = int(eti.get("count", 0))
#     eti_std  = float(eti.get("std_area_um2", 0.0))

#     plb_area = float(plb.get("total_area_um2", 0.0))
#     plb_cnt  = int(plb.get("count", 0))
#     plb_std  = float(plb.get("std_area_um2", 0.0))

#     pro_len  = float(pro.get("total_length_um", 0.0))
#     pro_cnt  = int(pro.get("count", 0))
#     pro_std  = float(pro.get("std_length_um", 0.0))

#     pg_diam  = float(pg.get("diameter_um", 0.0))   
#     pg_cnt   = int(pg.get("count", 0))
#     pg_std   = float(pg.get("std_diameter_um", 0.0))

#     sg_area  = float(sg.get("total_area_um2", 0.0))
#     sg_cnt   = int(sg.get("count", 0))
#     sg_std   = float(sg.get("std_area_um2", 0.0))

#     prompt = (
#         "You are a biology expert helping analyze plant cell ultrastructure segmentation data.\n\n"
#         f"- Etioplast Area (total): {eti_area:.3f} µm² ({eti_cnt} regions and {eti_std:.3f} µm² std)\n"
#         f"- PLB Area (total): {plb_area:.3f} µm² ({plb_cnt} regions and {plb_std:.3f} µm² std)\n"
#         f"- Prothylakoid Total Length: {pro_len:.3f} µm ({pro_cnt} regions and {pro_std:.3f} µm std)\n"
#         f"- Plastoglobule Avg. Diameter: {pg_diam:.3f} µm ({pg_cnt} regions and {pg_std:.3f} µm std)\n"
#         f"- Starch (total area): {sg_area:.3f} µm² ({sg_cnt} regions and {sg_std:.3f} µm² std)\n\n"
#         "Please summarize these findings in simple biological terms and explain what they might "
#         "suggest about the sample's plastid structure."
#     )
#     return prompt


# def get_generative_response(data):
#     prompt = generate_prompt(data)
#     payload = {
#         "model": MODEL,
#         "messages": [
#             {"role": "system", "content": "You are a helpful biology assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.7
#     }

#     response = requests.post(API_URL, headers=HEADERS, json=payload)

#     if response.status_code == 200:
#         return response.json()["choices"][0]["message"]["content"]
#     else:
#         raise Exception(f"LLM API error {response.status_code}: {response.text}")


# v1.0.2
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

def generate_prompt(analysis):
    # Safely extract data with fallbacks
    eti = analysis.get("Etioplast", {})
    plb = analysis.get("PLB", {})
    pro = analysis.get("Prothylakoid", {})
    pg  = analysis.get("Plastoglobule", {})
    sg  = analysis.get("StarchGrain", {}) # Fixed typo from StarchGain to StarchGrain

    prompt = (
        "You are an expert Plant Cell Biologist and Electron Microscopy Analyst specializing in plastid ultrastructure. "
        "I have provided quantitative segmentation data from an electron micrograph of a plant cell.\n\n"
        "### QUANTITATIVE DATA ###\n"
        f"- Etioplasts: {eti.get('count', 0)} detected | Total Area: {float(eti.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(eti.get('std_area_um2', 0.0)):.3f})\n"
        f"- Prolamellar Bodies (PLBs): {plb.get('count', 0)} detected | Total Area: {float(plb.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(plb.get('std_area_um2', 0.0)):.3f})\n"
        f"- Prothylakoids: {pro.get('count', 0)} detected | Total Length: {float(pro.get('total_length_um', 0.0)):.3f} µm (Std Dev: ±{float(pro.get('std_length_um', 0.0)):.3f})\n"
        f"- Plastoglobules: {pg.get('count', 0)} detected | Avg Diameter: {float(pg.get('mean_diameter_um', pg.get('diameter_um', 0.0))):.3f} µm (Std Dev: ±{float(pg.get('std_diameter_um', 0.0)):.3f})\n"
        f"- Starch Grains: {sg.get('count', 0)} detected | Total Area: {float(sg.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(sg.get('std_area_um2', 0.0)):.3f})\n\n"
        "### YOUR TASK ###\n"
        "Write a professional, cohesive laboratory conclusion (2-3 paragraphs) analyzing these metrics. \n"
        "Address the following biological implications:\n"
        "1. The developmental state of the plastid (e.g., skotomorphogenesis vs. readiness for photomorphogenesis based on PLB and Prothylakoid presence).\n"
        "2. The metabolic and physiological state (e.g., absence/presence of starch grains, implications of plastoglobule size/count regarding lipid storage or stress).\n"
        "3. Comment briefly on the consistency of the structures (using the Standard Deviation data).\n\n"
        "### FORMATTING CONSTRAINTS ###\n"
        "- Write in cohesive, flowing paragraphs.\n"
        "- DO NOT use numbered lists, bullet points, or markdown headers.\n"
        "- You may use **bold text** to highlight key metrics or structures, but keep formatting clean and simple."
    )
    return prompt


def get_generative_response(data):
    prompt = generate_prompt(data)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are a professional plant biologist writing clean, paragraph-based analytical reports. You never use bullet points."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5 # Lowered slightly for more factual, less hallucinated responses
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM API error {response.status_code}: {response.text}")