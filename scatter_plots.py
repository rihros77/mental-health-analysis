import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("survey_covid_merged.csv")

df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])
df['Time Period End Date'] = pd.to_datetime(df['Time Period End Date'])

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='covid_cases', y='Value', hue='Indicator')
plt.title("Survey Value vs COVID Cases")
plt.xlabel("COVID Cases")
plt.ylabel("Survey Value")
plt.legend(title="Indicator")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='covid_deaths', y='Value', hue='Indicator')
plt.title("Survey Value vs COVID Deaths")
plt.xlabel("COVID Deaths")
plt.ylabel("Survey Value")
plt.legend(title="Indicator")
plt.tight_layout()
plt.show()
