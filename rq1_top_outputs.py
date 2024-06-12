import json
import random

# This will pick a random selection of 33 of the combined most toxic outputs of the three models and give outputs of them in a text file

class result_set:
    id: int
    prompt: str
    score: float

    def __init__(self, id, prompt, score):
        self.id = id
        self.prompt = prompt
        self.score = score

    def __repr__(self):
        return str((self.id, self.prompt, self.score))


def main():
    print("eh")

    gemma_set_of_results = []
    llama_set_of_results = []
    mistral_set_of_results = []
    with open('gemma_scores.csv', encoding="utf8") as gemma, open('llama_scores.csv', encoding="utf8") as llama, open('mistral_scores.csv', encoding="utf8") as mistral:
        gemma_lines = gemma.readlines()
        llama_lines = llama.readlines()
        mistral_lines = mistral.readlines()

        for i in range(1, len(gemma_lines)):
            gemma_res = gemma_lines[i].split(',')
            gemma_prompt = ' '.join(gemma_res[1:-1])

            gemma_res_set = result_set(gemma_res[0], gemma_prompt, gemma_res[-1])
            gemma_set_of_results.append(gemma_res_set)

        for i in range(1, len(llama_lines)):
            llama_res = llama_lines[i].split(',')
            llama_prompt = ' '.join(llama_res[1:-1])

            llama_res_set = result_set(llama_res[0], llama_prompt, llama_res[-1])
            llama_set_of_results.append(llama_res_set)

        for i in range(1, len(gemma_lines)):
            mistral_res = mistral_lines[i].split(',')
            mistral_prompt = ' '.join(mistral_res[1:-1])

            mistral_res_set = result_set(mistral_res[0], mistral_prompt, mistral_res[-1])
            mistral_set_of_results.append(mistral_res_set)

    gemma.close()
    llama.close()
    mistral.close()

    gemmasort = sorted(gemma_set_of_results, key=lambda x: x.score, reverse=True)
    with open("gemma_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{gemmasort[i].id};{float(gemmasort[i].score)};{gemmasort[i].prompt}\n")

    llamasort = sorted(llama_set_of_results, key=lambda x: x.score, reverse=True)
    with open("llama_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{llamasort[i].id};{float(llamasort[i].score)};{llamasort[i].prompt}\n")

    mistralsort = sorted(mistral_set_of_results, key=lambda x: x.score, reverse=True)
    with open("mistral_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{mistralsort[i].id};{float(mistralsort[i].score)};{mistralsort[i].prompt}\n")

if __name__ == "__main__":
    main()