import jsonlines
import requests

#hugging_face_token = "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"

API_URL = "http://localhost:1234/v1/chat/completions"
headers = {"Authorization": "Bearer "}

# Define the system prompt and task description
task_desc = "Remember the system prompt instructions. Complete the following:"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def post_process_response(response, prompt_text):
    # Ensure the response is a list and get the generated text from the first item
    if isinstance(response, list) and len(response) > 0:
        generated_text = response['choices'][0]['message']['content']
        #clean_text = generated_text.replace(prompt_text, '').strip()
        return generated_text
    return ''

# Read prompts from the JSONL file
input_file = 'toxic.jsonl'
results = []


with jsonlines.open(input_file) as reader:
    data = {
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        "messages": [
            {"role": "user", "content": ""}
        ],
        "temperature": 0.8
    }
    for obj in reader:
        prompt_text = obj['prompt']['text']
        prompt = f"{task_desc}\n{prompt_text}"
        data["messages"][0]["content"] = prompt
        response = query(data)
        clean_response = post_process_response(response, prompt)
        print(clean_response)
        results.append({
            "filename": obj['filename'],
            "begin": obj['begin'],
            "end": obj['end'],
            "challenging": obj['challenging'],
            "prompt": obj['prompt'],
            "continuation": obj['continuation'],
            "response": clean_response
        })

# Save the results to a new JSONL file
output_file = 'toxic_outputs_llama_3_8b.jsonl'
with jsonlines.open(output_file, mode='w') as writer:
    writer.write_all(results)