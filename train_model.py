import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Обучаем ИИ на 10,000 ситуаций
np.random.seed(42)
data = []
for _ in range(10000):
    h = np.random.randint(0, 24)
    c = np.random.randint(0, 50) # Кол-во машин
    p = np.random.randint(0, 100) # Кол-во людей
    w = np.random.randint(0, 3) # Погода
    # Если ночь, яркость зависит от объектов. Если день — 0.
    b = (20 + c*1.5 + p*0.5 + w*10) if (h < 6 or h > 18) else 0
    data.append([h, c, p, w, min(max(b, 0), 100)])

df = pd.DataFrame(data, columns=['hour', 'cars', 'people', 'weather', 'brightness'])
model = GradientBoostingRegressor(n_estimators=100)
model.fit(df.drop('brightness', axis=1), df['brightness'])

with open('light_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Мозг обучен!")