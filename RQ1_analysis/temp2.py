import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

model = 'gemma'
path = 'conv_rq1_' + model + '_human_eval.csv'
data = pd.read_csv(path)
# Fit the model
# formula = ('LLMP_is_toxic ~ UP_is_toxic + UP_includes_racial_ref + UP_includes_gender_sexism + UP_includes_sexual_ref + UP_attack_character + UP_is_threat + UP_directed_at_ind_group + UP_political_mot + UP_allude_violence + UP_use_curse_word + UP_curse_word_context')
formula = ('LLMP_is_toxic ~ UP_includes_racial_ref + UP_includes_gender_sexism + UP_includes_sexual_ref + UP_is_threat')


model = smf.glm(formula, data=data).fit()

# Print the summary
print(model.summary())