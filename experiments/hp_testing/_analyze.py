"""Quick analysis of HP test results."""
import pandas as pd
df = pd.read_csv('experiments/hp_testing/results/stage_results_all.csv')
cols = ['test_regret_normalized', 'test_fairness', 'test_pred_mse']
summary = df.groupby(['method_label', 'lambda'])[cols].agg(['mean','std']).round(4)
summary.columns = [f'{c[0].replace("test_","")[:8]}_{c[1][:4]}' for c in summary.columns]
print(summary.to_string())
print()
saa_mse = df[df['method_label']=='SAA']['test_pred_mse'].mean()
print(f'SAA MSE baseline: {saa_mse:.4f}')
for m in ['FPTO','WDRO','FDFL-Scal','FDFL-PCGrad','FDFL-MGDA','FDFL-CAGrad']:
    sub = df[(df['method_label']==m) & (df['lambda']==0.0)]
    if sub.empty: continue
    mse = sub['test_pred_mse'].mean()
    reg = sub['test_regret_normalized'].mean()
    print(f'  {m:15s}: MSE={mse:.4f} ({(mse-saa_mse)/saa_mse*100:+.1f}% vs SAA), Regret={reg:.4f}')
