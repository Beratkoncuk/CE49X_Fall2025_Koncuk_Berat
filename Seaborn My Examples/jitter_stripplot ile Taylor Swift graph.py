import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Tema
sns.set_theme(style="whitegrid")

# --- Taylor Swift veri seti ---
np.random.seed(0)
eras = ["Fearless-era", "1989-era", "Folklore-era"]
measurements = ["tempo", "energy", "danceability", "valence"]

rows = []
for era in eras:
    for meas in measurements:
        # Her measurement için baz değer
        base = {"tempo": 80, "energy": 50, "danceability": 60, "valence": 50}[meas]
        # Era'lara küçük farklar ekle
        era_shift = {"Fearless-era": -5, "1989-era": 20, "Folklore-era": -10}[era]
        vals = np.random.normal(loc=base + era_shift, scale=5, size=40)
        for v in vals:
            rows.append({"era": era, "measurement": meas, "value": float(v)})

data = pd.DataFrame(rows)

# --- Grafik ---
f, ax = plt.subplots(figsize=(10, 6))
sns.despine(bottom=True, left=True)

# Kaç hue varsa ona göre dodge değeri
n_hue = data["era"].nunique()
dodge_width = 0.8 - 0.8 / n_hue

# Stripplot (gözlemler)
sns.stripplot(
    data=data,
    x="value", y="measurement", hue="era",
    dodge=True, jitter=0.25,
    alpha=0.55,  # görünürlük değiştirebiliyoruz burdan cnm 
    zorder=1, size=4, linewidth=0,
    palette="pastel",
    legend=False,
)

# Pointplot (ortalama değerler)
sns.pointplot(
    data=data,
    x="value", y="measurement", hue="era",
    dodge=dodge_width, palette="dark",
    errorbar=None,  # güven aralığı yok
    markers="D", markersize=8,
    linestyles="none", zorder=3,
)

# Legend ayarı
sns.move_legend(
    ax, loc="lower right", ncol=n_hue,
    frameon=True, columnspacing=1, handletextpad=0,
)

# Başlık ve eksen adları
ax.set_xlabel("Feature value (simulated)")
ax.set_title("Taylor Swift — Song feature observations & conditional means")

plt.tight_layout()
plt.show()
