SELECT
    age_group,
    AVG(actual_readmit) AS avg_readmit
FROM patient_risk_analysis
GROUP BY age_group;