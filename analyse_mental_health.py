import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_survey = pd.read_csv("anxiety_depression_survey.csv")
df_covid = pd.read_csv("weekly_covid_cases_deaths.csv")

print("Survey data sample:")
print(df_survey.head())
print("\nCOVID data sample:")
print(df_covid.head())

numeric_cols = ['tot_cases','new_cases','tot_deaths','new_deaths','new_historic_cases','new_historic_deaths']
for col in numeric_cols:
    df_covid[col] = df_covid[col].astype(str).str.replace(',','').astype(float)

df_survey['Time Period Start Date'] = pd.to_datetime(df_survey['Time Period Start Date'])
df_survey['Time Period End Date'] = pd.to_datetime(df_survey['Time Period End Date'])

df_covid['start_date'] = pd.to_datetime(df_covid['start_date'])
df_covid['end_date'] = pd.to_datetime(df_covid['end_date'])

us_state_abbrev = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
    'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri',
    'MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio',
    'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina',
    'SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont',
    'VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
}
df_covid['state_full'] = df_covid['state'].map(us_state_abbrev)

df_survey['State'] = df_survey['State'].str.lower()
df_covid['state_full'] = df_covid['state_full'].str.lower()

print("\nPreprocessed COVID data sample:")
print(df_covid.head())
print("\nPreprocessed Survey data sample:")
print(df_survey.head())

df_survey.to_csv("survey_preprocessed.csv", index=False)
df_covid.to_csv("covid_preprocessed.csv", index=False)
print("\nPreprocessed files saved as 'survey_preprocessed.csv' and 'covid_preprocessed.csv'.")
