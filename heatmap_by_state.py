import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("survey_covid_merged.csv")

heatmap_data = df.pivot_table(index='State', columns='Indicator', values='Value')

plt.figure(figsize=(12, 35))  
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".1f", cbar_kws={'shrink': 0.7})
plt.title("Survey Values by State and Indicator", fontsize=18)
plt.xlabel("Indicator", fontsize=14)
plt.ylabel("State", fontsize=14)
plt.yticks(fontsize=10)  
plt.xticks(rotation=45, fontsize=12)  
plt.tight_layout()
plt.show()
