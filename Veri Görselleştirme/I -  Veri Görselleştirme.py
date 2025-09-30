import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Jupyter için grafiklerin anında görünmesi
get_ipython().run_line_magic('matplotlib', 'inline')

# Veri seti
data = pd.read_csv('50_Startups.csv')

# İlk 5 satırı 
print(data.head())

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Veri setini yükle
data = pd.read_csv('50_Startups.csv')
print(data.head())

''' ~~~ 1. GÖREV: R&D Harcaması ve Kâr Arasındaki İlişki '''

plt.figure(figsize=(6,4))
plt.scatter(data["R&D Spend"], data["Profit"], color="blue")
plt.xlabel("R&D Harcaması")
plt.ylabel("Kâr")
plt.title("R&D Harcaması ile Kâr Arasındaki İlişki")
plt.show()

''' --- 2. GÖREV: Yönetim Harcaması ve Kâr Arasındaki İlişki '''
plt.figure(figsize=(6,4))
plt.scatter(data["Administration"], data["Profit"], color="green")
plt.xlabel("Yönetim Harcaması")
plt.ylabel("Kâr")
plt.title("Yönetim Harcaması ile Kâr Arasındaki İlişki")
plt.show()

''' >>> 3. GÖREV: Eyaletlere Göre Ortalama Kâr '''
state_profit = data.groupby("State")["Profit"].mean()

plt.figure(figsize=(6,4))
state_profit.plot(kind="bar", color=["orange", "purple", "red"])
plt.xlabel("Eyalet")
plt.ylabel("Ortalama Kâr")
plt.title("Eyaletlere Göre Ortalama Kâr")
plt.show()

''' *** 4. GÖREV: Harcama Türlerinin Dağılımı (Boxplot) '''
plt.figure(figsize=(6,4))
data[["R&D Spend", "Administration", "Marketing Spend"]].plot(kind="box")
plt.title("R&D, Yönetim ve Pazarlama Harcamalarının Dağılımı")
plt.ylabel("Harcama Tutarı")
plt.show()
