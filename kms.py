import pandas as pd

mistral_scores = pd.read_csv('mistral_scores_sorted.csv')

# mistral_scores.sort_values(by=['score'], ascending=False, inplace=True)



top30 = mistral_scores[0:30]
# print(top30.iloc[1]['score'])
with open("mistral_sort_2.csv", "w+") as output:
    output.write("id;toxicity_score;prompt;user_prompt\n")
    for i in range(30):
        output.write(f"{top30.iloc[i]['Unnamed: 0']};{float(top30.iloc[i]['score'])};{top30.iloc[i]['prompt']};{top30.iloc[i]['user_prompt']}\n")