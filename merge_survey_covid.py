import pandas as pd

# -----------------------------
# Load preprocessed CSVs
# -----------------------------
df_survey = pd.read_csv("survey_preprocessed.csv")
df_covid = pd.read_csv("covid_preprocessed.csv")

# Convert date columns to datetime
df_survey['Time Period Start Date'] = pd.to_datetime(df_survey['Time Period Start Date'])
df_survey['Time Period End Date'] = pd.to_datetime(df_survey['Time Period End Date'])
df_covid['start_date'] = pd.to_datetime(df_covid['start_date'])
df_covid['end_date'] = pd.to_datetime(df_covid['end_date'])

# -----------------------------
# Merge by State and Time Period
# -----------------------------
merged_data = []

for idx, row in df_survey.iterrows():
    mask = (
        (df_covid['state_full'] == row['State']) &
        (df_covid['start_date'] >= row['Time Period Start Date']) &
        (df_covid['end_date'] <= row['Time Period End Date'])
    )
    covid_sum = df_covid.loc[mask, ['new_cases', 'new_deaths']].sum()
    
    merged_data.append({
        **row.to_dict(),
        'covid_cases': covid_sum['new_cases'],
        'covid_deaths': covid_sum['new_deaths']
    })

df_merged = pd.DataFrame(merged_data)

# Quick check
print(df_merged.head())

# -----------------------------
# Save merged data (optional)
# -----------------------------
df_merged.to_csv("survey_covid_merged.csv", index=False)
print("Merged data saved as 'survey_covid_merged.csv'.")
