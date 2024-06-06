from googleapiclient import discovery
import json

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
  for text in texts:
    scores.append(float(evaluate_toxicity(text)))

  return scores
