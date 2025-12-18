import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

df = pd.read_csv('./mushra.csv')
examinee_num = len(df['session_uuid'].unique())
df = df[['trial_id', 'rating_stimulus', 'rating_score']]

# aggregate data
audiorefs = ['A', 'B', 'C', 'D', 'reference']

ant = [1, 2, 3]
james = [4, 5, 6]
meg = [7, 8, 9]

trialgroups = [ant, james, meg]
names = ['Ant', 'James', 'Megan']

trial_results: list[pd.DataFrame] = []
for t in range(0,3):
    trial_df = df.query(f'trial_id == "trial{trialgroups[t][0]}" | trial_id == "trial{trialgroups[t][1]}" | trial_id == "trial{trialgroups[t][2]}"')
    newdf = pd.DataFrame()
    for ref in audiorefs:
        ref_results = trial_df.query(f'rating_stimulus == "{ref}"')['rating_score'].values
        newdf.insert(0, ref, ref_results)
    trial_results.append(newdf)
    plt.figure()
    boxplot = trial_results[t].boxplot(column=audiorefs)
    plt.ylim([-10, 110])
    plt.title(f'{names[t]} Corpus MOS for Intelligibility')
    plt.savefig(f'./group{t}_results.png')

print(f'Number of responses: {examinee_num}')
