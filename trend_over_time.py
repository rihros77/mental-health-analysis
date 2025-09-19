import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("survey_covid_merged.csv")

df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])

plt.figure(figsize=(12,6))
df.groupby(['Time Period Start Date','Indicator'])['Value'].mean().unstack().plot()
plt.title("Average Survey Value Over Time")
plt.xlabel("Time Period")
plt.ylabel("Survey Value")
plt.legend(title="Indicator")
plt.tight_layout()
plt.show()
