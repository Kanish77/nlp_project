import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

# Load the CSV file
model_name = 'llama'
path = 'rq1_' + model_name + '_human_eval.csv'
data = pd.read_csv(path)

cat_cols = ['UP_is_toxic', 'UP_includes_racial_ref', 'UP_includes_gender_sexism', 'UP_includes_sexual_ref',
           'UP_attack_character', 'UP_is_threat', 'UP_directed_at_ind_group', 'UP_political_mot', 'UP_allude_violence',
           'UP_use_curse_word', 'UP_curse_word_context', 'LLMP_is_toxic', 'LLMP_includes_racial_ref',
           'LLMP_includes_gender_sexism','LLMP_includes_sexual_ref', 'LLMP_attack_character', 'LLMP_is_threat',
           'LLMP_directed_at_ind_group', 'LLMP_political_mot', 'LLMP_allude_violence', 'LLMP_use_curse_word',
           'LLMP_curse_word_context']
data[cat_cols] = data[cat_cols].astype("category")

# Descriptive statistics for categorical data
data_description = data.describe(include='all')
top_row = data_description.loc['top']

print("NOW PRINTING BASIC DESCRIPTIVE INFORMATION OF COLUMNS (MODE VALUES) \n")
print("------------------------------------------------------------------)")
for col in top_row.index:
    print(f'{col}: {top_row[col]}')
print("-------------------------------------------------------------------\n\n")

# function to perform Chi-Square test and display contingency table
def chi_square_test(column1, column2):
    contingency_table = pd.crosstab(data[column1], data[column2])
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    plt.figure(figsize=(12, 6))
    sns.heatmap(contingency_table, annot=True, cmap='Blues')
    plt.title(f'Contingency Table: {column1} vs {column2}')
    plt.xlabel(column2, fontsize=9)
    plt.ylabel(column1, fontsize=9)
    plt.yticks(rotation=45, fontsize=9)
    # plt.show()
    plt.savefig(f'{model_name}_{column1}_{column2}.png')
    return chi2, p

columns_to_test = [
    'UP_is_toxic',
    'UP_includes_racial_ref',
    'UP_includes_gender_sexism',
    'UP_includes_sexual_ref',
    'UP_attack_character',
    'UP_is_threat',
    'UP_directed_at_ind_group',
    'UP_political_mot',
    'UP_allude_violence',
    'UP_use_curse_word',
    'UP_curse_word_context'
]

# do Chi-Square tests
print("NOW DOING CHI-SQUARE TESTS BETWEEN CORRESPONDING COLUMNS")
print("------------------------------------------------------------------)")
chi_square_results = {}
for column in columns_to_test:
    chi2, p = chi_square_test(column, f'LLMP_{column.split("_", 1)[1]}')
    chi_square_results[column] = (chi2, p)
for col, res in chi_square_results.items():
    significant = res[1] < 0.05
    print("Test for", col, " Chi value: ", round(res[0],3), " p value: ", round(res[1], 3), "significant: ", significant)

print("-------------------------------------------------------------------\n\n")


print("NOW FITTING REGRESSION MODEL ON COLUMN VALUES TO SEE IF THERES ANY RELATIONSHIP")
# First convert categorical variables to numerical format
data['UP_is_toxic'] = data['UP_is_toxic'].map({'Yes': 1.0, 'No': 0.0, 'Somewhat': 0.75, 'Maybe?': 0.5})
data['LLMP_is_toxic'] = data['LLMP_is_toxic'].map({'Yes': 1.0, 'No': 0.0, 'Somewhat': 0.75, 'Maybe?': 0.5})

data['UP_includes_racial_ref'] = data['UP_includes_racial_ref'].map({'No': 0.0, 'Information': 0.25, 'Neutral': 0.5, 'Negative': 1.0, 'Slurs': 1.5})
data['LLMP_includes_racial_ref'] = data['LLMP_includes_racial_ref'].map({'No': 0.0, 'Information': 0.25, 'Neutral': 0.5, 'Negative': 1.0, 'Slurs': 1.5})

data['UP_includes_gender_sexism'] = data['UP_includes_gender_sexism'].map({'negative': 1, 'No': 0, 'Somewhat': 0.75, 'Informative': 0.25, 'Neutral': 0.5})
data['LLMP_includes_gender_sexism'] = data['LLMP_includes_gender_sexism'].map({'negative': 1, 'No': 0, 'Somewhat': 0.75, 'Informative': 0.25, 'Neutral': 0.5})

data['UP_includes_sexual_ref'] = data['UP_includes_sexual_ref'].map({'Yes (other)': 1, 'No': 0, 'kind of?': 0.6, 'Only in words, not context': 0.40, 'Used in sexual context': 0.8, 'Informative': 0.2})
data['LLMP_includes_sexual_ref'] = data['LLMP_includes_sexual_ref'].map({'Yes (other)': 1, 'No': 0, 'kind of?': 0.6, 'Only in words, not context': 0.40, 'Used in sexual context': 0.8, 'Informative': 0.2})

data['UP_attack_character'] = data['UP_attack_character'].map({'Yes': 1.0, 'Maybe?': 0.5, 'No': 0})
data['LLMP_attack_character'] = data['LLMP_attack_character'].map({'Yes': 1.0, 'Maybe?': 0.5, 'No': 0})

data['UP_is_threat'] = data['UP_is_threat'].map({'Yes': 1.0, 'Maybe?': 0.5, 'No': 0})
data['LLMP_is_threat'] = data['LLMP_is_threat'].map({'Yes': 1.0, 'Maybe?': 0.5, 'No': 0})

data['UP_directed_at_ind_group'] = data['UP_directed_at_ind_group'].map({'Yes': 1, 'No': 0, 'Maybe?': 0.5})
data['LLMP_directed_at_ind_group'] = data['LLMP_directed_at_ind_group'].map({'Yes': 1, 'No': 0, 'Maybe?': 0.5})

data['UP_political_mot'] = data['UP_political_mot'].map({'Directly political topic': 1, 'No': 0, 'Possibly': 0.5, 'Indirectly political topic': 0.75})
data['LLMP_political_mot'] = data['LLMP_political_mot'].map({'Directly political topic': 1, 'No': 0, 'Possibly': 0.5, 'Indirectly political topic': 0.75})

data['UP_allude_violence'] = data['UP_allude_violence'].map({'No': 0, 'Sarcastic/Joking manner': 0.17, 'Does so informatively': 0.34, 'In a general manner': 0.5, 'Yes (other)': 0.67, 'In a threatening manner': 1})
data['LLMP_allude_violence'] = data['LLMP_allude_violence'].map({'No': 0, 'Sarcastic/Joking manner': 0.17, 'Does so informatively': 0.34, 'In a general manner': 0.5, 'Yes (other)': 0.67, 'In a threatening manner': 1})

data['UP_use_curse_word'] = data['UP_use_curse_word'].map({'No': 0, 'Light use': 0.25, 'Moderate use': 0.5, 'Heavy use': 0.75, 'Very heavy use': 1.0})
data['LLMP_use_curse_word'] = data['LLMP_use_curse_word'].map({'No': 0, 'Light use': 0.25, 'Moderate use': 0.5, 'Heavy use': 0.75, 'Very heavy use': 1.0})

data['UP_curse_word_context'] = data['UP_curse_word_context'].map({'No': 0, 'Friendly/joking': 0.2, 'Casual use': 0.4, 'Emphasis/expression': 0.6, 'General insults': 0.8, 'Directed insults': 1.0})
data['LLMP_curse_word_context'] = data['LLMP_curse_word_context'].map({'No': 0, 'Friendly/joking': 0.2, 'Casual use': 0.4, 'Emphasis/expression': 0.6, 'General insults': 0.8, 'Directed insults': 1.0})

new_path = "conv_" + path
data.to_csv(new_path)