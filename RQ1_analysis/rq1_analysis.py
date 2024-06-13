import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the CSV file
data = pd.read_csv('rq1_gemma_human_eval.csv')


# function to perform Chi-Square test and display contingency table
def chi_square_test(column1, column2):
    contingency_table = pd.crosstab(data[column1], data[column2])
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    plt.figure(figsize=(10, 5))
    sns.heatmap(contingency_table, annot=True, cmap='Blues')
    plt.title(f'Contingency Table: {column1} vs {column2}')
    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.show()

    return chi2, p

columns_to_test = [
    'UP_is_toxic',
    'UP_includes_racial_ref',
    'UP_includes_gender_sexism',
    'UP_includes_sexual_ref',
    'UP_is_threat',
    'UP_directed_at_ind_group',
    'UP_political_mot',
    'UP_allude_violence',
    'UP_curse_word_context'
]

# do Chi-Square tests
chi_square_results = {}
for column in columns_to_test:
    chi2, p = chi_square_test(column, f'LLMP_{column.split("_", 1)[1]}')
    chi_square_results[column] = (chi2, p)

print(chi_square_results)
