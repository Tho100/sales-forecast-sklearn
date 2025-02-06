import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import StandardScaler

dates = pd.date_range(start='2024-01-01', end='2024-3-31', freq='D')

data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(10, 5000, len(dates))
})

data.sort_values(by='date', inplace=True)

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.weekday

X = data[['year', 'month', 'day', 'day_of_week']]
y = data['sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=12, test_size=0.5)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor())
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

future_dates = pd.date_range(
    start=data['date'].max() + pd.Timedelta(days=1), periods=120, freq='D'
) # 4 months range # Alternative for the next month: [data['date'].max() + datetime.timedelta(days=i) for i in range(1, 31)]

future_data = pd.DataFrame({
    'year': [date.year for date in future_dates],
    'month': [date.month for date in future_dates],
    'day': [date.day for date in future_dates],
    'day_of_week': [date.weekday() for date in future_dates],
})

future_predictions = pipeline.predict(future_data)

forecast_data = pd.DataFrame({
    'date': future_dates, 
    'predicted_sales': future_predictions
})

plt.style.use('seaborn')

plt.figure(figsize=(20, 6))

plt.plot(data['date'], data['sales'], label='Actual Sales', marker='o', color='red', alpha=.4)
plt.plot(forecast_data['date'], forecast_data['predicted_sales'], label='Sales Forecast', marker='o', color='skyblue', alpha=.6)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual Sales vs Forecast')
