import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

merged_file = r"C:\Users\rihro\OneDrive\Desktop\health-data\survey_covid_merged.csv"
df = pd.read_csv(merged_file)
df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])

df_agg = df.groupby(['Indicator','Time Period Start Date'])[['Value','covid_cases','covid_deaths']].mean().reset_index()

indicator_counts = df_agg['Indicator'].value_counts()
available_indicators = indicator_counts[indicator_counts >= 2].index.tolist()

if not available_indicators:
    raise ValueError("No indicator has enough data to fit Prophet model.")

indicator = available_indicators[0] 
print(f"Forecasting indicator: {indicator}")

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

future = m.make_future_dataframe(periods=12, freq='W')
future = future.merge(df_time[['ds','cases','deaths']], on='ds', how='left')
future['cases'] = future['cases'].ffill()
future['deaths'] = future['deaths'].ffill()

forecast = m.predict(future)

fig = m.plot(forecast)
plt.title(f"Forecast of {indicator}")
plt.tight_layout()
plt.show()
