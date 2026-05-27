SELECT
    risk_segment,
    AVG(actual_readmit) AS readmission_rate
FROM patient_risk_analysis
GROUP BY risk_segment;