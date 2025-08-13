# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==== Data Preparation ====
file_path = r"C:\Users\Dell\OneDrive\Desktop\SDA\Preprocessed_data (1).csv"
df = pd.read_csv(file_path)

input_cols = ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
              'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S',
               'T0pt150S', 'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S',
               'T1pt000S', 'T1pt200S', 'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']

df['Rjb_km'] = df['Rjb_km'].replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['Vs30_Selected_for_Analysis_m_s'] = df['Vs30_Selected_for_Analysis_m_s'].astype(float).replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['log_Rjb_km'] = np.log10(df['Rjb_km'])
df['log_Vs30_Selected_for_Analysis_m_s'] = np.log10(df['Vs30_Selected_for_Analysis_m_s'])

df = df.dropna(subset=input_cols + target_cols)

inputs = df[input_cols].values
targets = np.log(df[target_cols].values)  # log-transform targets

EqIDs = df['NGAsubEQID'].values
stratify_col = df['Intra_Inter_Flag'].values

scaler_input = StandardScaler()
scaler_target = StandardScaler()
inputs_scaled = scaler_input.fit_transform(inputs)
targets_scaled = scaler_target.fit_transform(targets)

X_train, X_temp, y_train, y_temp, EqID_train, EqID_temp, stratify_train, stratify_temp = train_test_split(
    inputs_scaled, targets_scaled, EqIDs, stratify_col, test_size=0.3, random_state=42, stratify=stratify_col)

X_val, X_test, y_val, y_test, EqID_val, EqID_test = train_test_split(
    X_temp, y_temp, EqID_temp, test_size=0.5, random_state=42, stratify=stratify_temp)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ==== Custom Bayesian Linear Layer ====
class CustomBayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-5)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.fill_(-5)

    def forward(self, x):
        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b
        return F.linear(x, weight, bias)

# ==== Bayesian Neural Network ====
class BNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = CustomBayesLinear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = CustomBayesLinear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = CustomBayesLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        self.out_mean = CustomBayesLinear(128, out_dim)
        self.out_logvar = CustomBayesLinear(128, out_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        mean = self.out_mean(x)
        logvar = self.out_logvar(x)
        return mean, logvar

# ==== Negative Log Likelihood Loss ====
def nll_loss(pred_mean, pred_logvar, target):
    var = F.softplus(pred_logvar) + 1e-6
    inv_var = 1.0 / var
    loss = torch.mean(inv_var * (target - pred_mean) ** 2 + torch.log(var))
    return loss

# ==== Training Function ====
def train(model, loader, optimizer, scheduler, epochs=1000):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred_mean, pred_logvar = model(xb)
            loss = nll_loss(pred_mean, pred_logvar, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

# ==== Model Initialization ====
base_model = BNN(X_train.shape[1], y_train.shape[1])
optimizer = optim.AdamW(base_model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=64, shuffle=True)

# ==== Training ====
train(base_model, train_loader, optimizer, scheduler, epochs=1000)

# ==== Prediction with Uncertainty ====
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    preds_var = []
    with torch.no_grad():
        for _ in range(samples):
            mean, logvar = model(X)
            preds_mean.append(mean.cpu().numpy())
            preds_var.append(np.exp(logvar.cpu().numpy()))
    preds_mean = np.stack(preds_mean)
    preds_var = np.stack(preds_var)

    mean_pred = preds_mean.mean(axis=0)
    epistemic_unc = preds_mean.var(axis=0)
    aleatory_unc = preds_var.mean(axis=0)
    return mean_pred, epistemic_unc, aleatory_unc

mean_pred, epistemic_unc, aleatory_unc = predict_with_uncertainty(base_model, X_test_tensor)

# ==== Compute overall R² score ====
overall_r2 = r2_score(Y_test_tensor.numpy(), mean_pred, multioutput='uniform_average')
print(f"Overall R² Score: {overall_r2:.4f}")

# ==== Scatter Plot ====
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_tensor.numpy().flatten(), mean_pred.flatten(), alpha=0.5, s=10)
min_val = min(Y_test_tensor.min(), mean_pred.min())
max_val = max(Y_test_tensor.max(), mean_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"BNN Prediction (R² = {overall_r2:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Save model ====
torch.save(base_model.state_dict(), 'base_bnn_model.pth')

# %%
# After you've already obtained predictions from your model:
# mean_pred, epistemic_unc, aleatory_unc = predict_with_uncertainty(...)

# Compute mean uncertainty values across all samples and targets
mean_epistemic = np.mean(epistemic_unc)
mean_aleatory = np.mean(aleatory_unc)
mean_total = mean_epistemic + mean_aleatory

# Print contribution of each
print(f"Average Epistemic Uncertainty: {mean_epistemic:.6f}")
print(f"Average Aleatory Uncertainty:  {mean_aleatory:.6f}")
print(f"Total Predictive Uncertainty:  {mean_total:.6f}")

# Percentage contribution
epistemic_pct = 100 * mean_epistemic / mean_total
aleatory_pct = 100 * mean_aleatory / mean_total

print(f"\nUncertainty Breakdown:")
print(f" - Epistemic: {epistemic_pct:.2f}%")
print(f" - Aleatory:  {aleatory_pct:.2f}%")


# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# === Period Labels to Numeric Periods ===
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S',
               'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 260.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Prepare input feature order ===
# ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km', 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))
loaded_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Build input feature vector (1 sample)
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform targets and exponentiate to get original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Plot spectral acceleration vs period
    ax.semilogy(periods, y_pred, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# === Period Labels to Numeric Periods ===
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S',
               'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 500.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Prepare input feature order ===
# ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km', 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))
loaded_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Build input feature vector (1 sample)
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform targets and exponentiate to get original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Plot spectral acceleration vs period
    ax.semilogy(periods, y_pred, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Intra-event: Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# === Period Labels to Numeric Periods ===
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S',
               'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 760.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Prepare input feature order ===
# ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km', 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))
loaded_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Build input feature vector (1 sample)
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform targets and exponentiate to get original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Plot spectral acceleration vs period
    ax.semilogy(periods, y_pred, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# === Period column names converted to numeric periods ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed parameters ===
fixed_mw = 7.0
fixed_vs30 = 260
fixed_depth = 30.0
log_vs30 = np.log10(max(fixed_vs30, 0.01))
intra_inter = 0  # Intra-event

# === Rjb values to vary ===
rjb_values = [50,80,100,130,170]
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # load your saved weights
loaded_model.eval()

# === Prediction function with uncertainty sampling ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return torch.tensor(mean), torch.tensor(std)

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(8, 6))

for i, rjb in enumerate(rjb_values):
    log_rjb = np.log10(max(rjb, 0.01))
    # Input features order: ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
    # 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
    X_input = np.array([[fixed_mw, rjb, log_rjb, log_vs30, intra_inter, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    mean_pred, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform target scaling and exponentiate to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred.numpy().reshape(1, -1)).flatten()
    y_pred = np.exp(y_pred_log)

    ax.plot(periods, y_pred,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markevery=5,
            linewidth=2,
            label=f'Rjb = {rjb} km')

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Rjb", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(title="Rjb (km)", fontsize=10, title_fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# === Period column names converted to numeric periods ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed parameters ===
fixed_mw = 7.0
fixed_vs30 = 260
fixed_depth = 30.0
log_vs30 = np.log10(max(fixed_vs30, 0.01))
intra_inter = 1  # Intra-event

# === Rjb values to vary ===
rjb_values = [50,80,100,130,170]
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # load your saved weights
loaded_model.eval()

# === Prediction function with uncertainty sampling ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return torch.tensor(mean), torch.tensor(std)

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(8, 6))

for i, rjb in enumerate(rjb_values):
    log_rjb = np.log10(max(rjb, 0.01))
    # Input features order: ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
    # 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
    X_input = np.array([[fixed_mw, rjb, log_rjb, log_vs30, intra_inter, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    mean_pred, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform target scaling and exponentiate to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred.numpy().reshape(1, -1)).flatten()
    y_pred = np.exp(y_pred_log)

    ax.plot(periods, y_pred,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markevery=5,
            linewidth=2,
            label=f'Rjb = {rjb} km')

# === Plot formatting ===
ax.set_title(" Spectral Acceleration vs period(Rjb Varying)", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(title="Rjb (km)", fontsize=10, title_fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# === Period column names converted to numeric periods ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed parameters ===
fixed_mw = 7.0
fixed_vs30 = 760
fixed_depth = 30.0
log_vs30 = np.log10(max(fixed_vs30, 0.01))
intra_inter = 0  # Intra-event

# === Rjb values to vary ===
rjb_values = [50,80,100,130,170]
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # load your saved weights
loaded_model.eval()

# === Prediction function with uncertainty sampling ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return torch.tensor(mean), torch.tensor(std)

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(8, 6))

for i, rjb in enumerate(rjb_values):
    log_rjb = np.log10(max(rjb, 0.01))
    # Input features order: ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
    # 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
    X_input = np.array([[fixed_mw, rjb, log_rjb, log_vs30, intra_inter, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    mean_pred, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform target scaling and exponentiate to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred.numpy().reshape(1, -1)).flatten()
    y_pred = np.exp(y_pred_log)

    ax.plot(periods, y_pred,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markevery=5,
            linewidth=2,
            label=f'Rjb = {rjb} km')

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Rjb", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(title="Rjb (km)", fontsize=10, title_fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# === Fixed Parameters ===
fixed_mw = 4
fixed_rjb = 20.0
log_rjb = np.log10(max(fixed_rjb, 0.01))
fixed_depth = 30.0
intra_inter_val = 1  # 0 means intra-event based on your data

# === Vs30 values to analyze ===
vs30_values = [200, 360, 550, 700]

# === Styles for plotting ===
styles = {
    200: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
    360: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5},
    550: {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5},
    700: {'color': '#d62728', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5},
}

# === Periods converted from target columns (your target_cols list) ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Load your trained model weights ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # Load weights saved from training
loaded_model.eval()

# === Prediction function (mean only) ===
def predict_mean_only(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    return mean

# === Plotting ===
fig, ax = plt.subplots(figsize=(9, 6))

for vs30 in vs30_values:
    log_vs30 = np.log10(max(vs30, 0.01))

    # Prepare input vector in order: [Earthquake_Magnitude, Rjb_km, log_Rjb_km,
    #                                 log_Vs30_Selected_for_Analysis_m_s, Intra_Inter_Flag, Hypocenter_Depth_km]
    X_input = np.array([[fixed_mw, fixed_rjb, log_rjb, log_vs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral acceleration
    mean_pred = predict_mean_only(loaded_model, X_tensor, samples=100)

    # Inverse transform (scaled target back to original log space)
    y_pred_log = scaler_target.inverse_transform(mean_pred.reshape(1, -1)).flatten()
    # Exponentiate to get back original scale (spectral acceleration)
    y_pred = np.exp(y_pred_log)

    # Plot SA vs Period
    ax.plot(periods, y_pred,
            label=f'Vs30 = {vs30} m/s',
            **styles[vs30],
            markersize=8,
            markevery=8)

# === Plot formatting ===
ax.set_title("Spectral Acceleration vs Period(Vs30 Varying) ", fontsize=14, weight='bold', pad=10)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(title="Vs30 (m/s)", fontsize=11, title_fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# === Fixed Parameters ===
fixed_mw = 5
fixed_rjb = 20.0
log_rjb = np.log10(max(fixed_rjb, 0.01))
fixed_depth = 30.0
intra_inter_val = 1 # 0 means intra-event based on your data

# === Vs30 values to analyze ===
vs30_values = [200, 360, 550, 700]

# === Styles for plotting ===
styles = {
    200: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
    360: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5},
    550: {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5},
    700: {'color': '#d62728', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5},
}

# === Periods converted from target columns (your target_cols list) ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Load your trained model weights ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # Load weights saved from training
loaded_model.eval()

# === Prediction function (mean only) ===
def predict_mean_only(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    return mean

# === Plotting ===
fig, ax = plt.subplots(figsize=(9, 6))

for vs30 in vs30_values:
    log_vs30 = np.log10(max(vs30, 0.01))

    # Prepare input vector in order: [Earthquake_Magnitude, Rjb_km, log_Rjb_km,
    #                                 log_Vs30_Selected_for_Analysis_m_s, Intra_Inter_Flag, Hypocenter_Depth_km]
    X_input = np.array([[fixed_mw, fixed_rjb, log_rjb, log_vs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral acceleration
    mean_pred = predict_mean_only(loaded_model, X_tensor, samples=100)

    # Inverse transform (scaled target back to original log space)
    y_pred_log = scaler_target.inverse_transform(mean_pred.reshape(1, -1)).flatten()
    # Exponentiate to get back original scale (spectral acceleration)
    y_pred = np.exp(y_pred_log)

    # Plot SA vs Period
    ax.plot(periods, y_pred,
            label=f'Vs30 = {vs30} m/s',
            **styles[vs30],
            markersize=8,
            markevery=8)

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Vs30", fontsize=14, weight='bold', pad=10)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(title="Vs30 (m/s)", fontsize=11, title_fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# === Fixed Parameters ===
fixed_mw = 5.5
fixed_rjb = 20.0
log_rjb = np.log10(max(fixed_rjb, 0.01))
fixed_depth = 30.0
intra_inter_val = 1  # 0 means intra-event based on your data

# === Vs30 values to analyze ===
vs30_values = [200, 360, 550, 700]

# === Styles for plotting ===
styles = {
    200: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
    360: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5},
    550: {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5},
    700: {'color': '#d62728', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5},
}

# === Periods converted from target columns (your target_cols list) ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Load your trained model weights ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # Load weights saved from training
loaded_model.eval()

# === Prediction function (mean only) ===
def predict_mean_only(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    return mean

# === Plotting ===
fig, ax = plt.subplots(figsize=(9, 6))

for vs30 in vs30_values:
    log_vs30 = np.log10(max(vs30, 0.01))

    # Prepare input vector in order: [Earthquake_Magnitude, Rjb_km, log_Rjb_km,
    #                                 log_Vs30_Selected_for_Analysis_m_s, Intra_Inter_Flag, Hypocenter_Depth_km]
    X_input = np.array([[fixed_mw, fixed_rjb, log_rjb, log_vs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral acceleration
    mean_pred = predict_mean_only(loaded_model, X_tensor, samples=100)

    # Inverse transform (scaled target back to original log space)
    y_pred_log = scaler_target.inverse_transform(mean_pred.reshape(1, -1)).flatten()
    # Exponentiate to get back original scale (spectral acceleration)
    y_pred = np.exp(y_pred_log)

    # Plot SA vs Period
    ax.plot(periods, y_pred,
            label=f'Vs30 = {vs30} m/s',
            **styles[vs30],
            markersize=8,
            markevery=8)

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Vs30", fontsize=14, weight='bold', pad=10)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(title="Vs30 (m/s)", fontsize=11, title_fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# === Custom Bayes Linear Layer ===
class CustomBayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-5)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.fill_(-5)

    def forward(self, x):
        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b
        return F.linear(x, weight, bias)

# === BNN ===
class BNN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.25):
        super().__init__()
        self.fc1 = CustomBayesLinear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = CustomBayesLinear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = CustomBayesLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_mean = CustomBayesLinear(128, out_dim)
        self.out_logvar = CustomBayesLinear(128, out_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        mean = self.out_mean(x)
        logvar = self.out_logvar(x)
        return mean, logvar

# === NLL Loss ===
def nll_loss(pred_mean, pred_logvar, target):
    var = F.softplus(pred_logvar) + 1e-6
    inv_var = 1.0 / var
    return torch.mean(inv_var * (target - pred_mean) ** 2 + torch.log(var))

# === Load and preprocess transfer dataset ===
file_path_new = r"C:\Users\Dell\OneDrive\Desktop\PSA_Indian_sub_with_logs.csv"
df = pd.read_csv(file_path_new)

input_cols = ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
              'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S',
               'T0pt150S', 'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S',
               'T1pt000S', 'T1pt200S', 'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']

df['Rjb_km'] = df['Rjb_km'].replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['Vs30_Selected_for_Analysis_m_s'] = df['Vs30_Selected_for_Analysis_m_s'].astype(float).replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['log_Rjb_km'] = np.log10(df['Rjb_km'])
df['log_Vs30_Selected_for_Analysis_m_s'] = np.log10(df['Vs30_Selected_for_Analysis_m_s'])

df = df.dropna(subset=input_cols + target_cols)

inputs = df[input_cols].values
targets = np.log(df[target_cols].values)

# === Same scaling logic ===
scaler_input = StandardScaler()
scaler_target = StandardScaler()
inputs_scaled = scaler_input.fit_transform(inputs)
targets_scaled = scaler_target.fit_transform(targets)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, targets_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# === Load model and freeze early layers ===
model = BNN(X_train.shape[1], y_train.shape[1], dropout_rate=0.1)
model.load_state_dict(torch.load("base_bnn_model.pth"))

for name, param in model.named_parameters():
    if "fc1" in name or "fc2" in name or "bn1" in name or "bn2" in name:
        param.requires_grad = False

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

# === Transfer Training ===
def train_transfer(model, loader, optimizer, scheduler, epochs=500):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred_mean, pred_logvar = model(xb)
            loss = nll_loss(pred_mean, pred_logvar, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

train_transfer(model, train_loader, optimizer, scheduler)

# === Predict with uncertainty ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean, preds_var = [], []
    with torch.no_grad():
        for _ in range(samples):
            mean, logvar = model(X)
            preds_mean.append(mean.cpu().numpy())
            preds_var.append(np.exp(logvar.cpu().numpy()))
    preds_mean = np.stack(preds_mean)
    preds_var = np.stack(preds_var)
    return preds_mean.mean(axis=0), preds_mean.var(axis=0), preds_var.mean(axis=0)

mean_pred, epistemic_unc, aleatory_unc = predict_with_uncertainty(model, X_test_tensor)

# === R² and per-target plots ===
print(f"Transfer R²: {r2_score(y_test, mean_pred):.4f}")
for i, name in enumerate(target_cols):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test[:, i], mean_pred[:, i], alpha=0.5, s=10)
    plt.plot([y_test[:, i].min(), y_test[:, i].max()],
             [y_test[:, i].min(), y_test[:, i].max()], 'r--')
    plt.xlabel(f"True (scaled) {name}")
    plt.ylabel(f"Predicted (scaled) {name}")
    plt.title(f"{name} (R² = {r2_score(y_test[:, i], mean_pred[:, i]):.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# === Period Labels to Numeric Periods ===
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S',
               'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 260.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Load your fine-tuned transfer model ===
transfer_model = BNN(X_train.shape[1], y_train.shape[1], dropout_rate=0.1)
transfer_model.load_state_dict(torch.load('base_bnn_model.pth'))  # if this path holds the *fine-tuned* weights
transfer_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Construct input vector with correct feature order
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean log spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(transfer_model, X_tensor, samples=100)

    # Inverse transform and exponentiate to get back to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Plotting
    ax.semilogy(periods, y_pred, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Intra-event: Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter1d

# === Period Labels to Numeric Periods ===
target_cols = [
    'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S',
    'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S', 'T0pt200S', 'T0pt300S',
    'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S',
    'T1pt200S', 'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S'
]
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 260.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Construct input vector with correct feature order
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean log spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(transfer_model, X_tensor, samples=100)

    # Inverse transform and exponentiate to get back to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Apply Gaussian smoothing filter
    sigma = 1  # Adjust sigma for more/less smoothing
    y_pred_smooth = gaussian_filter1d(y_pred, sigma=sigma)

    # Plotting smoothed curve
    ax.semilogy(periods, y_pred_smooth, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Transfer: Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter1d

# === Period Labels to Numeric Periods ===
target_cols = [
    'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S',
    'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S', 'T0pt200S', 'T0pt300S',
    'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S',
    'T1pt200S', 'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S'
]
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 760
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # 0 for intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Load your fine-tuned transfer model ===
# Assumes BNN class is defined and X_train, y_train, scaler_input, scaler_target are loaded

transfer_model = BNN(X_train.shape[1], y_train.shape[1], dropout_rate=0.1)
transfer_model.load_state_dict(torch.load('base_bnn_model.pth'))
transfer_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)  # Assumes model returns mean and uncertainty
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Prepare input vector for the model
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)  # scale inputs
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean and std dev for spectral accelerations in log scale
    mean_pred_log, std_pred_log = predict_with_uncertainty(transfer_model, X_tensor, samples=100)

    # Inverse transform from scaler and exp to original units
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Apply Gaussian smoothing filter to smooth the curve
    sigma = 0.7 # Adjust sigma for smoothing strength
    y_pred_smooth = gaussian_filter1d(y_pred, sigma=sigma)

    # Plot smoothed spectral acceleration vs period
    ax.semilogy(periods, y_pred_smooth, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Transfer: Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# === Period column names converted to numeric periods ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed parameters ===
fixed_mw = 7.0
fixed_vs30 = 760
fixed_depth = 30.0
log_vs30 = np.log10(max(fixed_vs30, 0.01))
intra_inter = 0  # Intra-event

# === Rjb values to vary ===
rjb_values = [50,80,100,130,170]
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # load your saved weights
loaded_model.eval()

# === Prediction function with uncertainty sampling ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return torch.tensor(mean), torch.tensor(std)

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(8, 6))

for i, rjb in enumerate(rjb_values):
    log_rjb = np.log10(max(rjb, 0.01))
    # Input features order: ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
    # 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
    X_input = np.array([[fixed_mw, rjb, log_rjb, log_vs30, intra_inter, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    mean_pred, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform target scaling and exponentiate to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred.numpy().reshape(1, -1)).flatten()
    y_pred = np.exp(y_pred_log)

    ax.plot(periods, y_pred,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markevery=5,
            linewidth=2,
            label=f'Rjb = {rjb} km')

# === Plot formatting ===
ax.set_title("Transfer: Spectral Acceleration vs Period(Rjb Varying) Sensitivity to Rjb", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(title="Rjb (km)", fontsize=10, title_fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Period column names converted to numeric periods ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed parameters ===
fixed_mw = 7.0
fixed_vs30 = 550
fixed_depth = 30.0
log_vs30 = np.log10(max(fixed_vs30, 0.01))
intra_inter = 1  # Intra-event

# === Rjb values to vary ===
rjb_values = [50, 80, 100, 130, 170]
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# === Load your trained model ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # load your saved weights
loaded_model.eval()

# === Prediction function with uncertainty sampling ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return torch.tensor(mean), torch.tensor(std)

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(8, 6))

for i, rjb in enumerate(rjb_values):
    log_rjb = np.log10(max(rjb, 0.01))
    # Input features order: ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
    # 'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
    X_input = np.array([[fixed_mw, rjb, log_rjb, log_vs30, intra_inter, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    mean_pred, _ = predict_with_uncertainty(loaded_model, X_tensor, samples=100)

    # Inverse transform target scaling and exponentiate to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred.numpy().reshape(1, -1)).flatten()
    y_pred = np.exp(y_pred_log)

    # Apply Savitzky-Golay filter for smoothing
    window_length = 7 if len(y_pred) >= 7 else (len(y_pred) // 2) * 2 + 1  # must be odd
    polyorder = 2
    y_pred_smooth = savgol_filter(y_pred, window_length=window_length, polyorder=polyorder)

    ax.plot(periods, y_pred_smooth,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markevery=5,
            linewidth=2,
            label=f'Rjb = {rjb} km')

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Rjb", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(title="Rjb (km)", fontsize=10, title_fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()


# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# === Fixed Parameters ===
fixed_mw = 4
fixed_rjb = 20.0
log_rjb = np.log10(max(fixed_rjb, 0.01))
fixed_depth = 30.0
intra_inter_val = 1  # 0 means intra-event based on your data

# === Vs30 values to analyze ===
vs30_values = [200, 360, 550, 700]

# === Styles for plotting ===
styles = {
    200: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
    360: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5},
    550: {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5},
    700: {'color': '#d62728', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5},
}

# === Periods converted from target columns (your target_cols list) ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Load your trained model weights ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # Load weights saved from training
loaded_model.eval()

# === Prediction function (mean only) ===
def predict_mean_only(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    return mean

# === Plotting ===
fig, ax = plt.subplots(figsize=(9, 6))

for vs30 in vs30_values:
    log_vs30 = np.log10(max(vs30, 0.01))

    # Prepare input vector in order: [Earthquake_Magnitude, Rjb_km, log_Rjb_km,
    #                                 log_Vs30_Selected_for_Analysis_m_s, Intra_Inter_Flag, Hypocenter_Depth_km]
    X_input = np.array([[fixed_mw, fixed_rjb, log_rjb, log_vs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral acceleration
    mean_pred = predict_mean_only(loaded_model, X_tensor, samples=100)

    # Inverse transform (scaled target back to original log space)
    y_pred_log = scaler_target.inverse_transform(mean_pred.reshape(1, -1)).flatten()
    # Exponentiate to get back original scale (spectral acceleration)
    y_pred = np.exp(y_pred_log)

    # Plot SA vs Period
    ax.plot(periods, y_pred,
            label=f'Vs30 = {vs30} m/s',
            **styles[vs30],
            markersize=8,
            markevery=8)

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Vs30", fontsize=14, weight='bold', pad=10)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(title="Vs30 (m/s)", fontsize=11, title_fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# === Fixed Parameters ===
fixed_mw = 5
fixed_rjb = 20.0
log_rjb = np.log10(max(fixed_rjb, 0.01))
fixed_depth = 30.0
intra_inter_val = 1  # 0 means intra-event based on your data

# === Vs30 values to analyze ===
vs30_values = [200, 360, 550, 800]

# === Styles for plotting ===
styles = {
    200: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
    360: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5},
    550: {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5},
    800: {'color': '#d62728', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5},
}

# === Periods converted from target columns (your target_cols list) ===
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Load your trained model weights ===
loaded_model = BNN(X_train.shape[1], y_train.shape[1])
loaded_model.load_state_dict(torch.load('base_bnn_model.pth'))  # Load weights saved from training
loaded_model.eval()

# === Prediction function (mean only) ===
def predict_mean_only(model, X, samples=100):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            mean_pred, _ = model(X)
            preds.append(mean_pred.cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    return mean

# === Plotting ===
fig, ax = plt.subplots(figsize=(9, 6))

for vs30 in vs30_values:
    log_vs30 = np.log10(max(vs30, 0.01))

    # Prepare input vector in order: [Earthquake_Magnitude, Rjb_km, log_Rjb_km,
    #                                 log_Vs30_Selected_for_Analysis_m_s, Intra_Inter_Flag, Hypocenter_Depth_km]
    X_input = np.array([[fixed_mw, fixed_rjb, log_rjb, log_vs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean spectral acceleration
    mean_pred = predict_mean_only(loaded_model, X_tensor, samples=100)

    # Inverse transform (scaled target back to original log space)
    y_pred_log = scaler_target.inverse_transform(mean_pred.reshape(1, -1)).flatten()
    # Exponentiate to get back original scale (spectral acceleration)
    y_pred = np.exp(y_pred_log)

    # Plot SA vs Period
    ax.plot(periods, y_pred,
            label=f'Vs30 = {vs30} m/s',
            **styles[vs30],
            markersize=8,
            markevery=8)

# === Plot formatting ===
ax.set_title("Intra-event: Sensitivity to Vs30", fontsize=14, weight='bold', pad=10)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(title="Vs30 (m/s)", fontsize=11, title_fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()



# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # intra-event flag
fixed_depth = 30
magnitudes = [4.5, 5.5, 6.5, 7.5,8.5]
rjb_values = np.linspace(1, 500, 200)

# Use the first 3 target columns only for plotting
target_subset = target_cols[:3]

# Set model to eval mode
base_model.eval()

# Store predictions in dict: mag -> target -> list of SA
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

for mag in magnitudes:
    for rjb in rjb_values:
        # Construct input feature vector with correct feature order
        input_features = np.array([[mag,
                                    rjb,
                                    np.log10(max(rjb, 0.01)),  # avoid log(0)
                                    np.log10(fixed_vs30),
                                    fixed_intra_inter,
                                    fixed_depth]])
        # Scale input
        input_scaled = scaler_input.transform(input_features)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict with MC sampling (100 samples) for mean prediction
        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)

        # Inverse scale to original log-scale SA
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()

        # Convert from log to linear scale
        sa_pred = np.exp(mean_pred_unscaled)

        # Save predictions for selected targets only
        for i, target in enumerate(target_cols):
            if target in target_subset:
                predictions[mag][target].append(sa_pred[i])

# --- Plotting ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    colors = ['red', 'blue', 'green']
    for idx, mag in enumerate(magnitudes):
        sa_vals = np.array(predictions[mag][target])
        plt.plot(rjb_values, sa_vals, label=f'Mw {mag}', color=colors[idx], linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # Intra-event flag
fixed_depth = 30.0
magnitudes = [4.5, 5.5, 6.5, 7.5, 8.5]
rjb_values = np.linspace(1, 500, 200)

# --- Target Columns (Ensure this is defined correctly) ---
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S',
               'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S',
               'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S', 'T1pt500S',
               'T2pt000S', 'T3pt000S', 'T4pt000S']

target_subset = target_cols[:3]  # First 3 target periods

# --- Ensure model is in eval mode ---
base_model.eval()

# --- Store predictions: {magnitude: {target_period: [SA values across Rjb]}} ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Loop over magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        # Input feature: [Mw, Rjb, logRjb, logVs30, IntraInterFlag, Depth]
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        # Scale and convert to tensor
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # MC Sampling for Bayesian prediction
        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)

        # Inverse scale and exponentiate
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)  # Convert log(SA) to SA

        # Store selected period predictions
        for i, target in enumerate(target_cols):
            if target in target_subset:
                predictions[mag][target].append(sa_values[i])

# --- Plotting ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])
        plt.plot(rjb_values, sa_vals, label=f'Mw {mag}', linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # Intra-event
fixed_depth = 30.0
magnitudes = [4.5,5.5, 6.5, 7.5,]
rjb_values = np.linspace(1, 500, 200)

# --- Subset of Periods (Target Columns) ---
target_subset = ['T0pt010S', 'T0pt020S', 'T0pt030S']  # You can change this
target_indices = [target_cols.index(t) for t in target_subset]

# --- Evaluate Mode ---
base_model.eval()

# --- Prediction Dictionary ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Loop over magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        # Construct input feature vector
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        # Scale input
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict with MC sampling
        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        
        # Mean prediction across samples
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)

        # Inverse transform from scaled log space → log SA → SA
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)

        # Store only subset values
        for i, target in zip(target_indices, target_subset):
            predictions[mag][target].append(sa_values[i])

# --- Plotting ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])
        plt.plot(rjb_values, sa_vals, label=f'Mw {mag}', linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # <-- Add this

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # Intra-event
fixed_depth = 30.0
magnitudes = [4.5, 5,5.5]
rjb_values = np.linspace(1, 500, 200)

# --- Subset of Periods (Target Columns) ---
target_subset = ['T0pt010S', 'T0pt020S', 'T0pt030S']  # You can change this
target_indices = [target_cols.index(t) for t in target_subset]

# --- Evaluate Mode ---
base_model.eval()

# --- Prediction Dictionary ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Loop over magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        # Construct input feature vector
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        # Scale input
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict with MC sampling
        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        
        # Mean prediction across samples
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)

        # Inverse transform from scaled log space → log SA → SA
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)

        # Store only subset values
        for i, target in zip(target_indices, target_subset):
            predictions[mag][target].append(sa_values[i])

# --- Plotting with Smoothing ---
from scipy.signal import savgol_filter

for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])

        # Apply Savitzky-Golay filter
        window_length = 11 if len(sa_vals) >= 11 else (len(sa_vals)//2)*2 + 1  # must be odd and <= len
        polyorder = 3 # can tweak if needed
        sa_smoothed = savgol_filter(sa_vals, window_length=window_length, polyorder=polyorder)

        plt.plot(rjb_values, sa_smoothed, label=f'Mw {mag}', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter  # <-- Import smoothing function

# === Period Labels to Numeric Periods ===
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt150S',
               'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T1pt200S',
               'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']
periods = [float(t[1:].replace("pt", ".").replace("S", "")) for t in target_cols]

# === Fixed Inputs ===
fixed_rjb = 20.0
fixed_vs30 = 760.0
logrjb = np.log10(max(fixed_rjb, 0.01))
logvs30 = np.log10(max(fixed_vs30, 0.01))

# === Magnitude Range ===
mw_range = [4.5, 5.5, 6.5, 7.5, 8.5]
intra_inter_val = 0  # Intra-event
fixed_depth = 30.0   # Hypocenter depth

# === Load your fine-tuned transfer model ===
transfer_model = BNN(X_train.shape[1], y_train.shape[1], dropout_rate=0.1)
transfer_model.load_state_dict(torch.load('base_bnn_model.pth'))  # fine-tuned model path
transfer_model.eval()

# === Prediction with uncertainty function ===
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    with torch.no_grad():
        for _ in range(samples):
            mean, _ = model(X)
            preds_mean.append(mean.cpu().numpy())
    preds_mean = np.stack(preds_mean)
    mean_pred = preds_mean.mean(axis=0)
    std_pred = preds_mean.std(axis=0)
    return torch.tensor(mean_pred), torch.tensor(std_pred)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

for mw in mw_range:
    # Construct input vector with correct feature order
    X_input = np.array([[mw, fixed_rjb, logrjb, logvs30, intra_inter_val, fixed_depth]])
    X_scaled = scaler_input.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict mean log spectral accelerations
    mean_pred_log, _ = predict_with_uncertainty(transfer_model, X_tensor, samples=100)

    # Inverse transform and exponentiate to get back to original scale
    y_pred_log = scaler_target.inverse_transform(mean_pred_log.numpy().reshape(1, -1))
    y_pred = np.exp(y_pred_log).flatten()

    # Apply Savitzky-Golay smoothing
    window_length = 5 if len(y_pred) >= 5 else (len(y_pred)//2)*2 + 1  # must be odd and <= len
    polyorder = 2
    y_pred_smooth = savgol_filter(y_pred, window_length=window_length, polyorder=polyorder)

    # Plotting
    ax.semilogy(periods, y_pred_smooth, label=f'Mw = {mw:.1f}', linewidth=2)

# === Plot formatting ===
ax.set_title("Intra-event: Spectral Acceleration vs Period (Magnitude Varying)", fontsize=14)
ax.set_xlabel("Period (s)", fontsize=12)
ax.set_ylabel("Spectral Acceleration (g)", fontsize=12)
ax.grid(True, which="both", linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.legend(title="Magnitude", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # Intra-event
fixed_depth = 30.0
magnitudes = [6,7,8]
rjb_values = np.linspace(1, 500, 200)

# --- Subset of Periods (Target Columns) ---
target_subset = ['T0pt010S', 'T0pt020S', 'T0pt030S']
target_indices = [target_cols.index(t) for t in target_subset]

# --- Evaluate Mode ---
base_model.eval()

# --- Prediction Dictionary ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Loop over magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        # Construct input feature vector
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        # Scale input
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict with MC sampling
        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        
        # Mean prediction across samples
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)

        # Inverse transform from scaled log space → log SA → SA
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)

        # Store only subset values
        for i, target in zip(target_indices, target_subset):
            predictions[mag][target].append(sa_values[i])

# --- Plotting Without Smoothing ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])
        plt.plot(rjb_values, sa_vals, label=f'Mw {mag}', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # <-- Gaussian filter

# --- Fixed Parameters ---
fixed_vs30 = 760
fixed_intra_inter = 0  # Intra-event
fixed_depth = 30.0
magnitudes = [6,7,8]
rjb_values = np.linspace(1, 500, 200)

# --- Subset of Periods (Target Columns) ---
target_subset = ['T0pt010S', 'T0pt020S', 'T0pt030S']
target_indices = [target_cols.index(t) for t in target_subset]

# --- Evaluate Mode ---
base_model.eval()

# --- Prediction Dictionary ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Loop over magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        preds = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, _ = base_model(input_tensor)
                preds.append(mean_pred.cpu().numpy())
        
        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0).reshape(1, -1)
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)

        for i, target in zip(target_indices, target_subset):
            predictions[mag][target].append(sa_values[i])

# --- Plotting with Gaussian Smoothing ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])

        # Apply Gaussian filter (sigma controls smoothing)
        sa_smoothed = gaussian_filter1d(sa_vals, sigma=2)  # You can tweak sigma

        plt.plot(rjb_values, sa_smoothed, label=f'Mw {mag}', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Fixed Parameters ---
fixed_vs30 = 500
fixed_intra_inter = 1  # Intra-event
fixed_depth = 30.0
magnitudes = [4.5,5,5.5,6]
rjb_values = np.linspace(1, 500, 200)

# --- Subset of Periods (Target Columns) ---
target_subset = ['T0pt020S']
target_indices = [target_cols.index(t) for t in target_subset]

# --- Prediction Dictionary ---
predictions = {mag: {target: [] for target in target_subset} for mag in magnitudes}

# --- Put model in evaluation mode ---
base_model.eval()

# --- Loop through magnitudes and Rjb values ---
for mag in magnitudes:
    for rjb in rjb_values:
        # Construct single input feature vector
        input_vector = np.array([[mag,
                                  rjb,
                                  np.log10(max(rjb, 0.01)),
                                  np.log10(fixed_vs30),
                                  fixed_intra_inter,
                                  fixed_depth]])
        
        # Apply input scaling
        input_scaled = scaler_input.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Monte Carlo sampling from BNN
        preds_samples = []
        with torch.no_grad():
            for _ in range(100):
                mean_pred, logvar_pred = base_model(input_tensor)
                # You could optionally sample from the predictive distribution using:
                # sampled_output = mean_pred + torch.sqrt(F.softplus(logvar_pred)) * torch.randn_like(mean_pred)
                preds_samples.append(mean_pred.cpu().numpy())

        preds_samples = np.stack(preds_samples)  # shape: (samples, 1, output_dim)
        mean_pred = preds_samples.mean(axis=0)   # shape: (1, output_dim)

        # Inverse transform: Scaled → Log(SA) → SA
        mean_pred_unscaled = scaler_target.inverse_transform(mean_pred).flatten()
        sa_values = np.exp(mean_pred_unscaled)  # Convert from log(SA) to SA

        # Store predictions for selected periods
        for idx, target in zip(target_indices, target_subset):
            predictions[mag][target].append(sa_values[idx])

# --- Plotting ---
for target in target_subset:
    plt.figure(figsize=(10, 7))
    
    for mag in magnitudes:
        sa_vals = np.array(predictions[mag][target])
        plt.plot(rjb_values, sa_vals, label=f'Mw {mag}', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Joyner-Boore Distance Rjb (km)', fontsize=13)
    plt.ylabel('Spectral Acceleration (g)', fontsize=13)
    plt.title(f'Spectral Acceleration vs Rjb for Period {target}', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==== Data Preparation ====
file_path = r"C:\Users\Dell\OneDrive\Desktop\SDA\Preprocessed_data (1).csv"
df = pd.read_csv(file_path)

input_cols = ['Earthquake_Magnitude', 'Rjb_km', 'log_Rjb_km',
              'log_Vs30_Selected_for_Analysis_m_s', 'Intra_Inter_Flag', 'Hypocenter_Depth_km']
target_cols = ['T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt040S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S',
               'T0pt150S', 'T0pt200S', 'T0pt300S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S',
               'T1pt000S', 'T1pt200S', 'T1pt500S', 'T2pt000S', 'T3pt000S', 'T4pt000S']

df['Rjb_km'] = df['Rjb_km'].replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['Vs30_Selected_for_Analysis_m_s'] = df['Vs30_Selected_for_Analysis_m_s'].astype(float).replace(0, 0.01).replace([np.inf, -np.inf], np.nan)
df['log_Rjb_km'] = np.log10(df['Rjb_km'])
df['log_Vs30_Selected_for_Analysis_m_s'] = np.log10(df['Vs30_Selected_for_Analysis_m_s'])

df = df.dropna(subset=input_cols + target_cols)

inputs = df[input_cols].values
targets = np.log(df[target_cols].values)  # log-transform targets

EqIDs = df['NGAsubEQID'].values
stratify_col = df['Intra_Inter_Flag'].values

scaler_input = StandardScaler()
scaler_target = StandardScaler()
inputs_scaled = scaler_input.fit_transform(inputs)
targets_scaled = scaler_target.fit_transform(targets)

X_train, X_temp, y_train, y_temp, EqID_train, EqID_temp, stratify_train, stratify_temp = train_test_split(
    inputs_scaled, targets_scaled, EqIDs, stratify_col, test_size=0.3, random_state=42, stratify=stratify_col)

X_val, X_test, y_val, y_test, EqID_val, EqID_test = train_test_split(
    X_temp, y_temp, EqID_temp, test_size=0.5, random_state=42, stratify=stratify_temp)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ==== Custom Bayesian Linear Layer ====
class CustomBayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-5)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.fill_(-5)

    def forward(self, x):
        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b
        return F.linear(x, weight, bias)

# ==== Bayesian Neural Network ====
class BNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = CustomBayesLinear(in_dim, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = CustomBayesLinear(8, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.25)
        self.out_mean = CustomBayesLinear(8, out_dim)
        self.out_logvar = CustomBayesLinear(8, out_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        mean = self.out_mean(x)
        logvar = self.out_logvar(x)
        return mean, logvar

# ==== Negative Log Likelihood Loss ====
def nll_loss(pred_mean, pred_logvar, target):
    var = F.softplus(pred_logvar) + 1e-6
    inv_var = 1.0 / var
    loss = torch.mean(inv_var * (target - pred_mean) ** 2 + torch.log(var))
    return loss

# ==== Training Function ====
def train(model, loader, optimizer, scheduler, epochs=1000):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred_mean, pred_logvar = model(xb)
            loss = nll_loss(pred_mean, pred_logvar, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

# ==== Model Initialization ====
base_model = BNN(X_train.shape[1], y_train.shape[1])
optimizer = optim.AdamW(base_model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=64, shuffle=True)

# ==== Training ====
train(base_model, train_loader, optimizer, scheduler, epochs=1000)

# ==== Prediction with Uncertainty ====
def predict_with_uncertainty(model, X, samples=100):
    model.eval()
    preds_mean = []
    preds_var = []
    with torch.no_grad():
        for _ in range(samples):
            mean, logvar = model(X)
            preds_mean.append(mean.cpu().numpy())
            preds_var.append(np.exp(logvar.cpu().numpy()))
    preds_mean = np.stack(preds_mean)
    preds_var = np.stack(preds_var)

    mean_pred = preds_mean.mean(axis=0)
    epistemic_unc = preds_mean.var(axis=0)
    aleatory_unc = preds_var.mean(axis=0)
    return mean_pred, epistemic_unc, aleatory_unc

mean_pred, epistemic_unc, aleatory_unc = predict_with_uncertainty(base_model, X_test_tensor)

# ==== Compute overall R² score ====
overall_r2 = r2_score(Y_test_tensor.numpy(), mean_pred, multioutput='uniform_average')
print(f"Overall R² Score: {overall_r2:.4f}")

# ==== Scatter Plot ====
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_tensor.numpy().flatten(), mean_pred.flatten(), alpha=0.5, s=10)
min_val = min(Y_test_tensor.min(), mean_pred.min())
max_val = max(Y_test_tensor.max(), mean_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"BNN Prediction (R² = {overall_r2:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Save model ====
torch.save(base_model.state_dict(), 'base_bnn_model.pth')


# %%
