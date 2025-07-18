import os
import re
import urllib.parse
import json
import difflib
import torch
import torch.nn.functional as F
import requests
import gradio as gr
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

LOGO_PATH = "logo.png"
API_KEY_ENVVAR = "OPENAI_API_KEY"
client = OpenAI(api_key=os.getenv(API_KEY_ENVVAR))

##########################################
# Severity Prediction using paragon-analytics/ADRv1 (Torch-only)
##########################################
severity_tokenizer = AutoTokenizer.from_pretrained("paragon-analytics/ADRv1")
severity_model = AutoModelForSequenceClassification.from_pretrained("paragon-analytics/ADRv1")

def adr_predict(text):
    """
    Computes the ADR severity score using the paragon-analytics/ADRv1 model.
    Tokenizes the input and returns a probability (float) for the severe ADR class.
    """
    encoded_input = severity_tokenizer(text, return_tensors="pt")
    output = severity_model(**encoded_input)
    scores = output.logits[0]
    prob = F.softmax(scores, dim=-1)
    return prob[1].item()

##########################################
# Function to fetch known adverse reactions for a medication using openFDA
##########################################
def get_medication_reactions(drug_name, limit=50):
    """
    Fetches all adverse reactions for a given medication using the openFDA Drug Event API.
    Returns a list of reaction terms (all in lowercase) extracted from the API response.
    """
    med_enc = urllib.parse.quote(drug_name)
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{med_enc}&limit={limit}"
    
    reactions = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            for reaction in result.get("patient", {}).get("reaction", []):
                term = reaction.get("reactionmeddrapt")
                if term:
                    reactions.append(term.lower())
    except Exception as e:
        print(f"FDA API request failed for {drug_name}: {e}")
    return list(set(reactions))

##########################################
# Function to extract medications and reactions using GPT4o via OpenAI API
##########################################
def extract_medications_and_reactions_gpt4o(text, access_token):
    """
    Uses GPT4o (via OpenAI's ChatCompletion API with model "gpt-4o") to extract medication names 
    and adverse reactions from the provided text. The prompt instructs GPT4o to account for casing issues,
    typos, non-medical references, and language variations, and to return its answer strictly as JSON with two keys:
    "medications" and "reactions". If nothing is found for a key, an empty list is returned.
    
    Parameters:
      - text: User-provided input text.
      - access_token: GPT4o API key. If left blank, the code will use the environment variable OPENAI_API_KEY.
    
    Returns:
      (medications, reactions): two lists of strings.
    """
    # If an access token is provided, use it; otherwise, fall back on OPENAI_API_KEY in the environment.
    token = access_token.strip() if access_token.strip() != "" else os.getenv("OPENAI_API_KEY")
    if not token:
        raise ValueError("No GPT4o access token provided and OPENAI_API_KEY is not set in the environment.")
    openai.api_key = token

    prompt = (
        "Extract the medication names and the adverse reactions mentioned in the following text. "
        "Account for possible casing issues, typos, non-medical references, and language variations. "
        "Return your answer strictly as JSON with two keys: \"medications\" and \"reactions\". "
        "For example: {\"medications\": [\"Tylenol\", \"Aspirin\"], \"reactions\": [\"headache\", \"nausea\"]}. "
        "If nothing is found for a key, return an empty list.\n\n"
        f"Text: {text}"
    )
    
    messages = [
        {"role": "system", "content": "You are an assistant that extracts medication names and adverse reactions from text."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=250,
            temperature=0.0
        )
    except Exception as e:
        raise ValueError(f"GPT4o API request failed: {e}")
    
    content = response.choices[0].message.content
    # Use a regex to isolate the first JSON object from the output.
    match = re.search(r'({.*?})', content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        raise ValueError(f"Could not extract JSON from GPT4o output: {content}")
    
    try:
        data = json.loads(json_str)
        medications = data.get("medications", [])
        reactions = data.get("reactions", [])
        if not isinstance(medications, list):
            medications = []
        if not isinstance(reactions, list):
            reactions = []
    except Exception as e:
        raise ValueError(f"Error parsing GPT4o output as JSON: {e}\nExtracted text: {json_str}") from e
    
    return medications, reactions

##########################################
# Function to compare extracted reactions with known reactions (using fuzzy matching)
##########################################
def compare_reactions(text_reactions, known_reactions, cutoff=0.8):
    """
    Compares the list of adverse reactions extracted from text with the list of known reactions
    (all normalized to lowercase). Fuzzy matching is used to account for minor typos or variations.
    
    Returns a list of reactions from text_reactions that are not found in known_reactions.
    """
    unknown_reactions = []
    known_set = set(kr.lower() for kr in known_reactions)
    for reaction in text_reactions:
        reaction_lower = reaction.lower()
        if reaction_lower in known_set:
            continue
        matches = difflib.get_close_matches(reaction_lower, list(known_set), n=1, cutoff=cutoff)
        if not matches:
            unknown_reactions.append(reaction)
    return unknown_reactions

##########################################
# Main processing function for the Gradio app
##########################################
def process_text(user_text, gpt4o_token):
    try:
        # 1. Extract medications and adverse reactions using GPT4o.
        medications, text_reactions = extract_medications_and_reactions_gpt4o(user_text, gpt4o_token)
    except ValueError as ve:
        return str(ve)
    
    # 2. For each extracted medication, fetch known adverse reactions from openFDA.
    known_reactions_all = {}
    combined_known_reactions = set()
    for med in medications:
        med_reactions = get_medication_reactions(med, limit=50)
        known_reactions_all[med] = med_reactions
        for r in med_reactions:
            combined_known_reactions.add(r)
    combined_known_reactions = list(combined_known_reactions)
    
    # 3. Compare the reactions extracted from text with the known reactions.
    new_reactions = compare_reactions(text_reactions, combined_known_reactions)
    
    # 4. Measure the severity of the reaction using your ADR model.
    severity_score = adr_predict(user_text)
    
    # 5. Compose the output summary.
    output = f"Severity Score: {severity_score:.2f}\n\n"
    output += "Extracted Medications:\n" + (", ".join(medications) if medications else "None") + "\n\n"
    output += "Extracted Reactions from Text:\n" + (", ".join(text_reactions) if text_reactions else "None") + "\n\n"
    output += "Known Reactions per Medication:\n"
    for med, known in known_reactions_all.items():
        output += f"  {med}: " + (", ".join(known) if known else "None") + "\n"
    output += "\nNew Reactions (in text but not in known data):\n" + (", ".join(new_reactions) if new_reactions else "None")
    
    return output

##########################################
# Gradio Interface Definition for Hugging Face Space App
##########################################
iface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter patient description…", label="Patient Text"),
        gr.Textbox(lines=1, placeholder="Enter GPT4o Access Token…", label="GPT4o Access Token")
    ],
    outputs=gr.Textbox(label="Analysis Summary"),
    title="Dr. Hubert's Medication, Reaction & Severity Analyzer (using GPT4o)",
    description=(
        # HTML image tag centered at the top
        f'<div align="center">'
        f'  <img src="{LOGO_PATH}" alt="App Logo" width="150" />'
        f'</div>\n\n'
        # then your normal prose
        "Provide text describing a patient's medication usage and reported adverse reactions. "
        "This app uses GPT4o (via the OpenAI API) to extract medications and reactions, "
        "fetches known adverse reactions for each medication from the openFDA API, "
        "compares them to identify new reactions, and calculates a severity score."
    )
)

##########################################
# Main execution for the Hugging Face Space App.
##########################################
if __name__ == "__main__":
    iface.launch()