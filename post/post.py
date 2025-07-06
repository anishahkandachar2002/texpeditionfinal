#for text check for toxicity using unitary/toxic-bert
#for image check for nudity using esvinj312/nudity-detection
#for videos split into frams and check
#for output use giving clip
#using regex for bad words or stop words for text 
#using spam detection whether it makes any false statemtn using 

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="unitary/toxic-bert")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

import os
from huggingface_hub import InferenceClient

# Paste your token here securely


client = InferenceClient(api_key=os.environ["HF_TOKEN"])

result = client.text_classification(
    "I kill you",
    model="unitary/toxic-bert"
)

print(result)
