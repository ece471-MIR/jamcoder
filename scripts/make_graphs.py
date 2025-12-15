import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

df = pd.read_csv('./results.csv')
examinee_num = len(df['session_uuid'].unique())
df = df[['trial_id', 'rating_stimulus', 'rating_score']]

# aggregate data
audiorefs = ['A', 'B', 'C', 'D', 'reference']

trial_results: list[pd.DataFrame] = []
for t in range(1,10):   # 9 trials
    trial_df = df.query(f'trial_id == "trial{t}"')
    newdf = pd.DataFrame()
    for ref in audiorefs:
        ref_results = trial_df.query(f'rating_stimulus == "{ref}"')['rating_score'].values
        newdf.insert(0, ref, ref_results)
    trial_results.append(newdf)
    plt.figure()
    boxplot = trial_results[t-1].boxplot(column=audiorefs)
    plt.ylim([0, 100])
    plt.title(f'Trial {t} MOS Results')
    plt.savefig(f'./trial{t}_results.png')

print(f'Number of responses: {examinee_num}')
