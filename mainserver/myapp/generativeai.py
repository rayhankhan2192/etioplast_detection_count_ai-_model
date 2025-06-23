# generativeai.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_prompt(data):
    prompt = (
        f"You are a biology expert helping analyze plant cell ultrastructure segmentation data. "
        f"Here is the analysis:\n\n"
        f"- Etioplast Area: {data['Etioplast']['area_um2']} µm² ({data['Etioplast']['count']} regions)\n"
        f"- PLB Area: {data['PLB']['area_um2']} µm² ({data['PLB']['count']} regions)\n"
        f"- Prothylakoid Total Length: {data['Prothylakoid']['total_length_um']} µm "
        f"({data['Prothylakoid']['count']} regions)\n"
        f"- Plastoglobule Avg. Diameter: {data['Plastoglobule']['avg_diameter_um']} µm "
        f"({data['Plastoglobule']['count']} regions)\n\n"
        f"Please summarize these findings in simple biological terms and explain what they might suggest about the sample's plastid structure."
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
