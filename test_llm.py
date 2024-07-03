# openai-env\Scripts\activate to activate virtual environment (note for claire)

# Supervised-learning model for the LLM using a generated set of user questions & answers. 
# Still in the exploration stage & testing how to use the OpenAI API!
# I used this template (https://colab.research.google.com/drive/16e_HJwCHNsATFVtljZTMg0JydrHAhO8A) as reference. 

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


# Setting dataset prompt.
prompt = "A conversational assistant that is helping users complete design tasks in Fusion 360, in the form of asking open-ended questions about loads, materials, obstacle geometry, and structure design."
temperature = .7
number_of_examples = 50


# Generates a dataset. 
N_RETRIES = 3
@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]
    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# Generate examples.
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)
print(prev_examples)


# Generating system message.
def generate_system_message(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message.content
system_message = generate_system_message(prompt)
print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')


# Puts generated examples into a dataframe and turns them into a final pair of datasets.
# Initialize lists to store prompts and responses
prompts = []
responses = []
# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass
# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})
# Remove duplicates
df = df.drop_duplicates()
print('There are ' + str(len(df)) + ' successfully-generated examples.')
# Initialize list to store training examples
training_examples = []
# Create training examples in the format required for GPT-3.5 fine-tuning
for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }
    training_examples.append(training_example)
# Save training examples to a .jsonl file
with open('training_examples.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')


# Uploads file to OpenAI.
file_id = openai.files.create(
  file=open("/content/training_examples.jsonl", "rb"),
  purpose='fine-tune'
).id

# Trains model.
job = openai.fine_tuning.jobs.create(training_file=file_id, model="gpt-4o")
job_id = job.id


# Creates fine-tuned model.
openai.fine_tuning.jobs.list_events(job_id, limit=10)


# Gets fine-tuned model name.
model_name_pre_object = openai.fine_tuning.jobs.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)


# Tests the model!
response = openai.chat.completions.create(
    model=model_name,
    messages=[
      {
        "role": "system",
        "content": system_message,
      },
      {
          "role": "user",
          "content": df['prompt'].sample().values[0],
      }
    ],
)
msg = response.choices[0].message.content
print(msg)