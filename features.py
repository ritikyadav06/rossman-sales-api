import numpy as np
import pandas as pd

def build_features(requests, stores_raw, store_day_avg, columns):

  df = requests.copy().reset_index(drop=True)

  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
  df['Year'] = df['Date'].dt.year
  df['Month'] = df['Date'].dt.month
  df['Day'] = df['Date'].dt.day
  df['DayOfWeek'] = df["Date"].dt.dayofweek + 1

  df = df.merge(stores_raw, on='Store', how='left')
  df = df.merge(store_day_avg, on=['Store', 'DayOfWeek'], how='left')

  store_mean = store_day_avg.groupby('Store')["StoreDayAvgSales"].mean()
  missing = df['StoreDayAvgSales'].isna()
  df.loc[missing, 'StoreDayAvgSales'] = df.loc[missing, 'Store'].map(store_mean)

  df['StateHoliday'] = df['StateHoliday'].astype(str)
  df['PromoInterval'] = df['PromoInterval'].fillna('None').astype(str)
  fill_values = {
    'CompetitionDistance' : 0,
    'CompetitionOpenSinceMonth' : 0,
    'CompetitionOpenSinceYear' : 0,
    'Promo2SinceWeek' : 0,
    'Promo2SinceYear' : 0
  }
  df.fillna(value=fill_values, inplace=True)

  # Derived features
  df['CompetitionOpenDurationMonths'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
  df.loc[df['CompetitionOpenSinceYear'] == 0, 'CompetitionOpenDurationMonths'] = 0
  df['CompetitionOpenDurationMonths'] = df['CompetitionOpenDurationMonths'].apply(lambda x: x if x > 0 else 0)

  df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
  df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
  df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
  df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

  month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
  df['MonthName'] = df['Month'].map(month_map)
  def is_promo2_active(row):
      if row['Promo2'] == 0 or row['PromoInterval'] == 'None':
          return 0
      # Check if current month is in the PromoInterval (e.g. 'Jan,Apr,Jul,Oct')
      active_months = [m.strip() for m in row['PromoInterval'].split(',')]
      return 1 if row['MonthName'] in active_months else 0
  df['IsPromo2ActiveMonth'] = df.apply(is_promo2_active, axis=1)

  return df[columns]
