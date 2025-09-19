import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("survey_covid_merged.csv")
df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])

# ---------------- Correlation ----------------
corr_cases = df['covid_cases'].corr(df['Value'])
corr_deaths = df['covid_deaths'].corr(df['Value'])
print("Correlation (cases vs survey):", corr_cases)
print("Correlation (deaths vs survey):", corr_deaths)

# ---------------- State Rankings ----------------
state_means = df.groupby("State")['Value'].mean().sort_values(ascending=False)
print("\nTop 5 States:\n", state_means.head(5))
print("\nBottom 5 States:\n", state_means.tail(5))

# ---------------- Pre vs Post Vaccine ----------------
pre_vaccine = df[df['Time Period Start Date'] < "2021-01-01"]
post_vaccine = df[df['Time Period Start Date'] >= "2021-01-01"]
print("\nPre-vaccine avg:", pre_vaccine['Value'].mean())
print("Post-vaccine avg:", post_vaccine['Value'].mean())

# ---------------- Boxplot ----------------
df['Period'] = df['Time Period Start Date'].apply(lambda x: "Pre-Vaccine" if x < pd.Timestamp("2021-01-01") else "Post-Vaccine")
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Period", y="Value")
plt.title("Survey Values Before vs After Vaccine Rollout")
plt.show()

# ---------------- Top/Bottom 5 States Bar Chart ----------------
top5 = state_means.head(5)
bottom5 = state_means.tail(5)
pd.concat([top5, bottom5]).plot(kind='bar', color='skyblue', figsize=(10,6))
plt.title("Top & Bottom 5 States - Avg Survey Values")
plt.ylabel("Average Survey Value")
plt.show()
