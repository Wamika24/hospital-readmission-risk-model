import pandas as pd

risk = pd.read_csv('fact_patient_risk.csv')
risk.insert(0, 'patient_id', range(1, len(risk)+1))
risk.to_csv('fact_patient_risk.csv', index=False)

print('Done. Rows:', len(risk))
print('First column:', risk.columns[0])
print('First 3 patient IDs:', list(risk['patient_id'].head(3)))