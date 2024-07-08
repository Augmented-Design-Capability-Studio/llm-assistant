# openai-env\Scripts\activate to activate virtual environment (note for claire)

# Practicing pulling from the OpenAI API. Hopefully should be less expensive. 

# Packages imported.
import os
from openai import OpenAI
import random
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import pandas as pd
from dotenv import load_dotenv


# API key
load_dotenv()
openai = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)


# Setting dataset prompt & requirements.
promptFile = open("prompt.txt", "rt") 
prompt = promptFile.read()         
promptFile.close()
model_temperature = .7
model_max_tokens = 256


response = openai.chat.completions.create(
  model = "gpt-4o",
  temperature = model_temperature,
  max_tokens = model_max_tokens,
  messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "Hello!"}
  ]
)

msg = response.choices[0].message.content

print(msg)