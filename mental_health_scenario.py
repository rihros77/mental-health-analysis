import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

merged_file = r"C:\Users\rihro\OneDrive\Desktop\health-data\survey_covid_merged.csv"
df = pd.read_csv(merged_file)
df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])


df_agg = df.groupby(['Indicator','Time Period Start Date'])[['Value','covid_cases','covid_deaths']].mean().reset_index()


indicator_counts = df_agg['Indicator'].value_counts()
available_indicators = indicator_counts[indicator_counts >= 2].index.tolist()
if not available_indicators:
    raise ValueError("No indicator has enough data to fit Prophet model.")

indicator = available_indicators[0] 
print(f"Running scenario analysis for indicator: {indicator}")

df_time = df_agg[df_agg['Indicator'] == indicator].copy()

df_time.rename(columns={'Time Period Start Date':'ds',
                        'Value':'y',
                        'covid_cases':'cases',
                        'covid_deaths':'deaths'}, inplace=True)

df_time['y'] = df_time['y'].ffill()
df_time['cases'] = df_time['cases'].fillna(0)
df_time['deaths'] = df_time['deaths'].fillna(0)

m = Prophet()
m.add_regressor('cases')
m.add_regressor('deaths')
m.fit(df_time[['ds','y','cases','deaths']])

future = m.make_future_dataframe(periods=12, freq='W')  # forecast 12 weeks ahead
future = future.merge(df_time[['ds','cases','deaths']], on='ds', how='left')
future['cases'] = future['cases'].ffill()
future['deaths'] = future['deaths'].ffill()

future_scenario = future.copy()
future_scenario['cases'] = future_scenario['cases'] * 1.5
future_scenario['deaths'] = future_scenario['deaths'] * 1.2

forecast_baseline = m.predict(future)
forecast_scenario = m.predict(future_scenario)

plt.figure(figsize=(12,6))
plt.plot(forecast_baseline['ds'], forecast_baseline['yhat'], label='Baseline', color='blue')
plt.plot(forecast_scenario['ds'], forecast_scenario['yhat'], label='Scenario: 50% surge', color='red', linestyle='--')
plt.fill_between(forecast_scenario['ds'], forecast_scenario['yhat_lower'], forecast_scenario['yhat_upper'],
                 color='red', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Survey Value (Mental Health Indicator)')
plt.title(f"Mental Health Forecast: Baseline vs Surge Scenario ({indicator})")
plt.legend()
plt.tight_layout()
plt.show()

output_file = r"C:\Users\rihro\OneDrive\Desktop\health-data\scenario_forecast_{0}.csv".format(indicator.replace(' ','_').lower())
forecast_scenario[['ds','yhat','yhat_lower','yhat_upper']].to_csv(output_file, index=False)
print(f"Scenario forecast saved to {output_file}")
