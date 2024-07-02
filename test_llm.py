# openai-env\Scripts\activate to activate virtual environment

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# test example from openai

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are an assistant tasked with helping designers model a program in Autodesk Fusion 360."}, 
    {"role": "user", "content": "Hello! Could you tell me how to design a bracket for a ship engine?"} 
  ]
)

msg = completion.choices[0].message.content

print(msg)