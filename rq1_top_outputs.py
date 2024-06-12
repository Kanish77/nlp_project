import json
import random

# This will pick a random selection of 33 of the combined most toxic outputs of the three models and give outputs of them in a text file

class result_set:
    id: int
    gemma_prompt: str
    llama_prompt: str
    mistral_prompt: str
    gemma_score: float
    llama_score: float
    mistral_score: float

    def __init__(self, id, gemma_prompt, llama_prompt, mistral_prompt, gemma_score, llama_score, mistral_score):
        self.id = id
        self.gemma_prompt= gemma_prompt
        self.llama_prompt = llama_prompt
        self.mistral_prompt = mistral_prompt
        self.gemma_score = gemma_score
        self.llama_score = llama_score
        self.mistral_score = mistral_score

    def __repr__(self):
        return str((self.id, self.gemma_prompt, self.llama_prompt, self.mistral_prompt, self.gemma_score, self.llama_score, self.llama_score))


def main():
    print("eh")

    set_of_results = []
    with open('gemma_scores.csv', encoding="utf8") as gemma, open('llama_scores.csv', encoding="utf8") as llama, open('mistral_scores.csv', encoding="utf8") as mistral:
        gemma_lines = gemma.readlines()
        llama_lines = llama.readlines()
        mistral_lines = mistral.readlines()

        for i in range(1, len(gemma_lines)):
            gemma_res = gemma_lines[i].split(',')
            llama_res = llama_lines[i].split(',')
            mistral_res = mistral_lines[i].split(',')

            gemma_prompt = ' '.join(gemma_res[1:-1])
            llama_prompt = ' '.join(llama_res[1:-1])
            mistral_prompt = ' '.join(llama_res[1:-1])
            res_set = result_set(gemma_res[0], gemma_prompt, llama_prompt, mistral_prompt, gemma_res[-1], llama_res[-1], mistral_res[-1])
            set_of_results.append(res_set)

    gemma.close()
    llama.close()
    mistral.close()

    gemmasort = sorted(set_of_results, key=lambda x: x.gemma_score, reverse=True)
    with open("gemma_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{gemmasort[i].id};{float(gemmasort[i].gemma_score)};{gemmasort[i].gemma_prompt}\n")

    llamasort = sorted(set_of_results, key=lambda x: x.llama_score, reverse=True)
    with open("llama_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{llamasort[i].id};{float(llamasort[i].llama_score)};{llamasort[i].llama_prompt}\n")

    mistralsort = sorted(set_of_results, key=lambda x: x.mistral_score, reverse=True)
    with open("mistral_top_100.csv", "w+") as output:
        output.write("id;toxicity_score;prompt\n")
        for i in range(100):
            output.write(f"{mistralsort[i].id};{float(mistralsort[i].mistral_score)};{mistralsort[i].mistral_prompt}\n")

if __name__ == "__main__":
    main()