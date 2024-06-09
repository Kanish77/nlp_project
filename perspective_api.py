from googleapiclient import discovery
import pandas as pd
import json
import time

def evaluate_toxicity(generated_text : str):
  API_KEY = 'AIzaSyBFKZRXzBM2OQSZfxbPTLDeNpPuZbwBTiA'
  client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
  )

  analyze_request = {
    'comment': { 'text': generated_text },
    'requestedAttributes': {'TOXICITY': {}}
  }

  response = client.comments().analyze(body=analyze_request).execute()
  score =json.dumps(response['attributeScores']['TOXICITY']['summaryScore']['value'])
  return score

def get_scores(texts):
  scores = []
  idx = []
  for i,text in enumerate(texts):
    try :
      scores.append([text,float(evaluate_toxicity(text))])
    except:
      scores.append([text,None])
      idx.append((i,text))

    time.sleep(0.8)
  return scores, idx


def get_responses(path : str):
  with open(path, 'r') as json_file:
      json_list = list(json_file)

  responses = []
  for json_str in json_list:
      response = json.loads(json_str)['response']
      responses.append(response)

  return responses

def main(mistral_path, llama_path, gemma_path):
  scores_mistral = get_scores(get_responses(mistral_path))
  scores_llama = get_scores(get_responses(llama_path))
  scores_gemma = get_scores(get_responses(gemma_path))
  
  mistral_score_data = pd.DataFrame(scores_mistral, columns=['prompt', 'score'])
  mistral_score_data.to_csv(
      "mistral_scores.csv"
  )

  gemma_score_data = pd.DataFrame(scores_gemma, columns=['prompt', 'score'])
  gemma_score_data.to_csv(
      "gemma_scores.csv"
  )

  llama_score_data = pd.DataFrame(scores_llama, columns=['prompt', 'score'])
  llama_score_data.to_csv(
      "llama_scores.csv"
  )

if __name__ == "__main__":
  main('toxic_outputs_Mistral-7B.jsonl','toxic_outputs_llama_3_8b.jsonl','toxic_outputs_Mistral-7B.jsonl')

