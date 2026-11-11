"""Arsitektur jaringan saraf. HARUS sama dengan yang dipakai saat training di Google Colab."""
import torch.nn as nn

# Urutan fitur input (harus sama dengan notebook)
FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain", "Gender_Male",
]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]


class ChurnNet(nn.Module):
    def __init__(self, n_in: int = len(FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)  # keluaran berupa logit; probabilitas = sigmoid(logit)
