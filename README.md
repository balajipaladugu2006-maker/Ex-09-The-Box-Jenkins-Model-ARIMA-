# Ex-09-The-Box-Jenkins-Model-ARIMA-
## NAME : Paladugu Venkata Balaji
## REG NO : 2305001024
## Aim
To understand and implement the Box–Jenkins approach to time-series forecasting using an ARIMA model in Python.

## Objectives
To create and visualize monthly sales time-series data.
To test stationarity using the Augmented Dickey-Fuller (ADF) test.
To apply differencing and analyze ACF and PACF plots.
To fit an ARIMA model and compare AIC and BIC values.
To perform residual diagnostic checking.
To forecast future sales and visualize the predicted values.
## Program
~~~python 
# Step 1: Import the required libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Step 2: Create sample monthly sales data
sales = [
    100, 105, 103, 110, 115, 112,
    120, 125, 123, 130, 135, 132,
    140, 145, 143, 150, 155, 152,
    160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("STEP 1: Original Sales Data")
print(data.to_string(index=False))

# Step 3: Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Test stationarity using the ADF test
adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))

if adf_result[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Insufficient evidence of stationarity.")

# Step 5: Apply first differencing
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Test stationarity after differencing
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))

if adf_diff[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Stationarity is not confirmed.")

# Step 7: Plot ACF and PACF
print("\nSTEP 5: ACF and PACF Plots")

plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()

plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()

# Step 8: Fit the ARIMA model
# Example candidate: ARIMA(1,1,1)
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

# Step 9: Display AIC and BIC
print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))

# Step 10: Diagnostic checking of residuals
residuals = fitted_model.resid

plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()

# Step 11: Perform the Ljung-Box test
lb_result = acorr_ljungbox(
    residuals,
    lags=[5],
    return_df=True
)

print("\nLjung-Box Test Result")
print(lb_result)

if lb_result["lb_pvalue"].iloc[0] > 0.05:
    print("No significant residual autocorrelation detected at lag 5.")
else:
    print("Residual autocorrelation may remain; review the model.")

# Step 12: Forecast the next six months
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")

for month, value in enumerate(forecast_mean, start=25):
    print(f"Month {month}: {value:.2f}")

# Display forecast confidence intervals
print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))

# Step 13: Plot actual data and forecast
future_months = range(len(data), len(data) + 6)

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(data)), data,
    marker="o", label="Actual Sales"
)
plt.plot(
    future_months, forecast_mean.to_numpy(),
    marker="o", linestyle="--", label="Forecast"
)

plt.fill_between(
    future_months,
    forecast_ci.iloc[:, 0].to_numpy(),
    forecast_ci.iloc[:, 1].to_numpy(),
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month (0-based index)")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 14: Display the final result
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed.")
print("Identification, estimation, diagnostics, and forecasting performed.")

~~~
## Output

<img width="712" height="716" alt="image" src="https://github.com/user-attachments/assets/3e869bf1-9f07-415c-b278-3f1a0eb825b2" />
<img width="1072" height="417" alt="image" src="https://github.com/user-attachments/assets/dba9a3b3-a762-4714-8111-4e526822ede1" />
<img width="1082" height="345" alt="image" src="https://github.com/user-attachments/assets/c2132bb2-7438-4ae9-8d03-a59a76c31486" />
<img width="558" height="691" alt="image" src="https://github.com/user-attachments/assets/8e8247df-3ad7-43eb-891c-cc5a55fce204" />
<img width="1085" height="793" alt="image" src="https://github.com/user-attachments/assets/34d911f3-1298-49cf-b633-9eac15b4b35c" />
<img width="793" height="578" alt="image" src="https://github.com/user-attachments/assets/b5e446ee-44fc-4b3a-aecf-16e6915b4da8" />
<img width="891" height="596" alt="image" src="https://github.com/user-attachments/assets/d8a62d16-dc9e-4a1d-b2d4-3aa9c3390a46" />
<img width="1083" height="437" alt="image" src="https://github.com/user-attachments/assets/eab46e1a-c483-4252-8602-aec47a264fc6" />
<img width="1067" height="545" alt="image" src="https://github.com/user-attachments/assets/135770a6-9469-48ec-9e5a-ce7aeb8b4909" />
<img width="872" height="696" alt="image" src="https://github.com/user-attachments/assets/084688a3-7f72-4427-999c-387909d650e3" />
<img width="651" height="440" alt="image" src="https://github.com/user-attachments/assets/aef75f8f-d567-4c79-8c87-54e177eeb27b" />
<img width="1080" height="537" alt="image" src="https://github.com/user-attachments/assets/56916200-265c-44ef-95bb-860d2c3dd4d8" />

## Result 
The Box–Jenkins approach was successfully implemented in Python using the ARIMA model. Stationarity testing, differencing, ACF/PACF analysis, model estimation, residual diagnostic checking, and six-month sales forecasting were performed.



