import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# 📌 Exempeldata
# max_lutning, riskprocent, längd, svårighetsgrad (etikett)
data = pd.DataFrame({
    "lutning": [1, 3, 5, 7, 10],
    "risk": [5, 10, 20, 30, 50],
    "längd": [500, 1000, 1500, 2000, 3000],
    "klass": ["lätt", "lätt", "medel", "svår", "svår"]
})

# 🔢 Omvandla etiketter till siffror
label_map = {"lätt": 0, "medel": 1, "svår": 2}
data["klass_num"] = data["klass"].map(label_map)

# 📊 Träna modellen
X = data[["lutning", "risk", "längd"]]
y = data["klass_num"]
model = DecisionTreeClassifier()
model.fit(X, y)

# 💾 Spara modellen och label_map
joblib.dump({"model": model, "label_map": label_map}, "ml_modell.pkl")

print("Modell sparad som 'ml_modell.pkl'")
