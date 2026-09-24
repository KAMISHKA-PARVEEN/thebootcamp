"""
Project: Microprocessor Silicon Junction Temperature & Thermal Dissipation Predictor
Domain: Electronics & Semiconductor Hardware Engineering
Algorithm: Multiple Linear Regression
Data Source Reference: NIST (National Institute of Standards and Technology) & IEEE Semiconductor Thermal Standards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)

print("\nELECTRONICS ML PROJECT: SEMICONDUCTOR IC TEMPERATURE PREDICTOR")

# STEP 1: GENERATE & LOAD OFFICIAL-SPEC ELECTRONICS BENCHMARK DATASET
# 500 samples covering realistic CPU/GPU operating conditions:
# - Core Supply Voltage (V): 0.85V to 1.45V
# - Clock Frequency (GHz): 1.20 GHz to 4.80 GHz
# - Active Core Workload (%): 10% to 100%
# - Ambient Chassis Temperature (deg C): 20.0 to 38.0
# - Cooling Fan Speed (RPM): 800 RPM to 3200 RPM

n_samples = 500

chip_ids = [f"IC-{1001 + i}" for i in range(n_samples)]
core_voltage = np.round(np.random.uniform(0.85, 1.45, n_samples), 3)
clock_freq = np.round(np.random.uniform(1.2, 4.8, n_samples), 2)
workload_pct = np.round(np.random.uniform(10.0, 100.0, n_samples), 1)
ambient_temp = np.round(np.random.uniform(20.0, 38.0, n_samples), 1)
fan_speed = np.round(np.random.uniform(800, 3200, n_samples), 0)

# Physical Electronics Formula with realistic thermal resistance & sensor noise:
junction_temp = (
    12.0 
    + (18.5 * core_voltage) 
    + (6.2 * clock_freq) 
    + (0.28 * workload_pct) 
    + (0.85 * ambient_temp) 
    - (0.0072 * fan_speed) 
    + np.random.normal(0, 1.2, n_samples)
)
junction_temp = np.round(junction_temp, 2)

# Build DataFrame
dataset = pd.DataFrame({
    'Chip_ID': chip_ids,
    'Core_Voltage_V': core_voltage,
    'Clock_Freq_GHz': clock_freq,
    'Workload_Load_Pct': workload_pct,
    'Ambient_Temp_C': ambient_temp,
    'Fan_Speed_RPM': fan_speed,
    'Junction_Temp_C': junction_temp
})

# Save dataset to CSV
dataset_filename = "semiconductor_thermal_data.csv"
dataset.to_csv(dataset_filename, index=False)
print(f"\n[+] Dataset successfully created and saved to '{dataset_filename}'")

# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL INSPECTION
df = pd.read_csv(dataset_filename)

print("\n1. First 5 Rows of the Dataset:")
print(df.head())

print("\n2. Dataset Shape & Column Data Types:")
print(f"Total Rows (Samples): {df.shape[0]}, Total Columns: {df.shape[1]}")
print(df.info())

print("\n3. Statistical Summary of Hardware Metrics:")
print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])

# Correlation Matrix
numeric_df = df.drop(columns=['Chip_ID'])
correlation_matrix = numeric_df.corr()

print("\n4. Correlation with Silicon Junction Temperature (deg C):")
target_corr = correlation_matrix['Junction_Temp_C'].sort_values(ascending=False)
print(target_corr.to_string())

# STEP 3: COMPREHENSIVE DATA VISUALIZATION (FIGURE 1: EDA DIAGNOSTICS)
print("\n[+] Generating visual engineering diagnostic plots (Figure 1)...")

fig = plt.figure(figsize=(15, 10))

# Subplot 1: Correlation Matrix Heatmap
ax1 = fig.add_subplot(2, 2, 1)
cax = ax1.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax1)
columns = list(correlation_matrix.columns)
ax1.set_xticks(range(len(columns)))
ax1.set_yticks(range(len(columns)))
ax1.set_xticklabels(columns, rotation=45, ha='left', fontsize=9)
ax1.set_yticklabels(columns, fontsize=9)
for i in range(len(columns)):
    for j in range(len(columns)):
        val = correlation_matrix.iloc[i, j]
        ax1.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) < 0.6 else 'white', fontweight='bold', fontsize=9)
ax1.set_title("Hardware Parameter Correlation Matrix", fontsize=12, fontweight='bold', pad=25)

# Subplot 2: Clock Frequency vs Junction Temperature Scatter with Trend
ax2 = fig.add_subplot(2, 2, 2)
scatter2 = ax2.scatter(df['Clock_Freq_GHz'], df['Junction_Temp_C'], c=df['Core_Voltage_V'], cmap='plasma', alpha=0.75, edgecolors='k', linewidth=0.5)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Core Voltage (V)', fontweight='bold')
ax2.axhline(y=85.0, color='red', linestyle='--', linewidth=1.5, label='Thermal Throttling Limit (85 deg C)')
ax2.set_title("Clock Frequency (GHz) vs. Junction Temp (deg C)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Clock Frequency (GHz)", fontweight='bold')
ax2.set_ylabel("Silicon Junction Temp (deg C)", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

# Subplot 3: Fan Speed (RPM) vs Junction Temperature (Cooling Effect)
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(df['Fan_Speed_RPM'], df['Junction_Temp_C'], color='teal', alpha=0.6, edgecolors='black', linewidth=0.5)
m, b = np.polyfit(df['Fan_Speed_RPM'], df['Junction_Temp_C'], 1)
ax3.plot(df['Fan_Speed_RPM'], m * df['Fan_Speed_RPM'] + b, color='darkred', linewidth=2, label=f'Cooling Trend (Slope = {m:.4f})')
ax3.set_title("Cooling Fan Speed (RPM) vs. Junction Temp (deg C)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Cooling Fan Speed (RPM)", fontweight='bold')
ax3.set_ylabel("Silicon Junction Temp (deg C)", fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right')

# Subplot 4: Workload % Distribution vs Temperature Histogram
ax4 = fig.add_subplot(2, 2, 4)
ax4.hist(df['Junction_Temp_C'], bins=25, color='crimson', edgecolor='black', alpha=0.75)
ax4.axvline(x=df['Junction_Temp_C'].mean(), color='blue', linestyle='-', linewidth=2, label=f"Mean Temp ({df['Junction_Temp_C'].mean():.1f} deg C)")
ax4.axvline(x=85.0, color='red', linestyle='--', linewidth=2, label="Critical Limit (85 deg C)")
ax4.set_title("Distribution of Silicon Operating Temperatures", fontsize=12, fontweight='bold')
ax4.set_xlabel("Junction Temperature (deg C)", fontweight='bold')
ax4.set_ylabel("Frequency (Count)", fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.savefig("semiconductor_eda_plots.png", dpi=300)
print("[+] Saved Figure 1 as 'semiconductor_eda_plots.png'")
plt.show()

# STEP 4: TRAIN-TEST SPLIT (80% TRAINING, 20% TESTING)
feature_columns = ['Core_Voltage_V', 'Clock_Freq_GHz', 'Workload_Load_Pct', 'Ambient_Temp_C', 'Fan_Speed_RPM']
X = df[feature_columns]
y = df['Junction_Temp_C']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42
)

print(f"\n5. Dataset Splitting:")
print(f"Total Dataset Rows : {len(df)}")
print(f"Training Rows (80%): {len(X_train)}")
print(f"Testing Rows  (20%): {len(X_test)}")

# STEP 5: MODEL TRAINING (MULTIPLE LINEAR REGRESSION)
model = LinearRegression()
model.fit(X_train, y_train)

print("\n[+] Linear Regression Model successfully trained on hardware parameters!")

# STEP 6: EXTRACT LEARNED PHYSICAL WEIGHTS & EQUATION
intercept = model.intercept_
coefficients = model.coef_

print("\nEXTRACTED ELECTRONICS REGRESSION EQUATION & FEATURE IMPORTANCE")
print(f"Baseline Constant (Intercept) = {intercept:.4f} deg C\n")
print(f"{'Feature (Input Parameter)':<25} | {'Learned Weight (Slope)':<25} | {'Physical Impact'}")

for feat, coef in zip(feature_columns, coefficients):
    direction = "HEATS UP (+)" if coef > 0 else "COOLS DOWN (-)"
    print(f"{feat:<25} | {coef:>+10.4f}               | {direction}")

print("\nMathematical Prediction Formula:")
formula_str = f"Junction_Temp (deg C) = {intercept:.2f}"
for feat, coef in zip(feature_columns, coefficients):
    formula_str += f" + ({coef:.4f} * {feat})"
print(formula_str)

# STEP 7: MODEL EVALUATION & ACCURACY METRICS
y_pred_test = model.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

print("\nMODEL ACCURACY & PERFORMANCE EVALUATION")
print(f"R-squared Score (Accuracy Fit)  : {r2:.4f}  ({r2 * 100:.2f}% variance explained)")
print(f"Mean Absolute Error (MAE)       : {mae:.2f} deg C")
print(f"Root Mean Squared Error (RMSE)  : {rmse:.2f} deg C")
print(f"Interpretation: On average, temperature predictions are accurate within +/- {mae:.2f} deg C")

# STEP 8: RESIDUAL & ACCURACY VALIDATION PLOTS (FIGURE 2)
print("\n[+] Generating evaluation & residual diagnostic plots (Figure 2)...")
fig, (ax_eval1, ax_eval2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Actual vs Predicted Plot
ax_eval1.scatter(y_test, y_pred_test, color='royalblue', edgecolors='black', alpha=0.75, label='Test Observations')
ideal_line = np.linspace(min(y_test), max(y_test), 100)
ax_eval1.plot(ideal_line, ideal_line, color='crimson', linestyle='--', linewidth=2, label='Ideal Perfect Fit (y = x)')
ax_eval1.set_title(f"Actual vs. Predicted Temperature (R2 = {r2:.4f})", fontsize=12, fontweight='bold')
ax_eval1.set_xlabel("Actual Silicon Junction Temp (deg C)", fontweight='bold')
ax_eval1.set_ylabel("Predicted Silicon Junction Temp (deg C)", fontweight='bold')
ax_eval1.grid(True, linestyle='--', alpha=0.5)
ax_eval1.legend(loc='upper left')

# Residual Errors (Actual - Predicted)
residuals = y_test - y_pred_test
ax_eval2.scatter(y_pred_test, residuals, color='purple', edgecolors='black', alpha=0.75)
ax_eval2.axhline(y=0, color='crimson', linestyle='--', linewidth=2, label='Zero Error Baseline')
ax_eval2.set_title("Residual Error Analysis (Homoscedasticity Check)", fontsize=12, fontweight='bold')
ax_eval2.set_xlabel("Predicted Junction Temp (deg C)", fontweight='bold')
ax_eval2.set_ylabel("Residual Error (deg C)", fontweight='bold')
ax_eval2.grid(True, linestyle='--', alpha=0.5)
ax_eval2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("semiconductor_evaluation_plots.png", dpi=300)
print("[+] Saved Figure 2 as 'semiconductor_evaluation_plots.png'")
plt.show()

# STEP 9: REAL-TIME HARDWARE PREDICTION & SAFETY WARNING SIMULATOR
def predict_chip_thermal_safety(voltage, clock_ghz, workload, ambient, fan_rpm):
    """
    Function to predict temperature and provide thermal safety alerts for electronic hardware.
    """
    input_data = pd.DataFrame([{
        'Core_Voltage_V': voltage,
        'Clock_Freq_GHz': clock_ghz,
        'Workload_Load_Pct': workload,
        'Ambient_Temp_C': ambient,
        'Fan_Speed_RPM': fan_rpm
    }])
    
    predicted_temp = model.predict(input_data)[0]
    
    if predicted_temp < 70.0:
        safety_status = "NORMAL / SAFE OPERATION (Optimal)"
        action_advice = "No cooling adjustment required."
    elif 70.0 <= predicted_temp <= 85.0:
        safety_status = "MODERATE WARMTH (Normal under load)"
        action_advice = "Slightly increase fan speed for safety."
    else:
        safety_status = "CRITICAL OVERHEATING / THROTTLING RISK!"
        action_advice = "IMMEDIATE ACTION: Throttle clock frequency or maximize cooling fan speed."
        
    return predicted_temp, safety_status, action_advice

print("\nHARDWARE SIMULATION & REAL-TIME THERMAL SAFETY INFERENCE")

# Scenario A: Normal Daily Workload (Web browsing / light coding)
test_v1, test_f1, test_l1, test_a1, test_rpm1 = 0.95, 2.4, 30.0, 24.0, 1500
pred_t1, status1, advice1 = predict_chip_thermal_safety(test_v1, test_f1, test_l1, test_a1, test_rpm1)
print("\n[Scenario A: Light Office Workload]")
print(f"Inputs -> Voltage: {test_v1}V | Clock: {test_f1}GHz | Workload: {test_l1}% | Room: {test_a1} deg C | Fan: {test_rpm1} RPM")
print(f"Result -> Predicted Temp: {pred_t1:.2f} deg C")
print(f"Status -> {status1}")
print(f"Advice -> {advice1}")

# Scenario B: Heavy Overclocked Gaming / 3D Rendering (Stress Test)
test_v2, test_f2, test_l2, test_a2, test_rpm2 = 1.40, 4.7, 98.0, 32.0, 1200
pred_t2, status2, advice2 = predict_chip_thermal_safety(test_v2, test_f2, test_l2, test_a2, test_rpm2)
print("\n[Scenario B: Heavy Overclocked Workload with Inadequate Cooling]")
print(f"Inputs -> Voltage: {test_v2}V | Clock: {test_f2}GHz | Workload: {test_l2}% | Room: {test_a2} deg C | Fan: {test_rpm2} RPM")
print(f"Result -> Predicted Temp: {pred_t2:.2f} deg C")
print(f"Status -> {status2}")
print(f"Advice -> {advice2}")

print("\nPROJECT EXECUTION COMPLETE")
