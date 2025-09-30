import pandas as pd

df = pd.read_csv("country.csv", decimal=".", thousands=",")
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = df[col].str.replace(",", ".").astype(float)
        except:
            pass

# 1. Nüfusa göre azalan sırada ülkeler
pop_desc = df.sort_values(by="Population", ascending=False)[["Country", "Population"]]

# 2. GDP per capita’ya göre artan sırada ülkeler
gdp_asc = df.sort_values(by="GDP ($ per capita)", ascending=True)[["Country", "GDP ($ per capita)"]]

# 3. Nüfusu 10 milyonun üzerinde olan ülkeler
pop_over_10m = df[df["Population"] > 10_000_000][["Country", "Population"]]

# 4. Literacy oranına göre en yüksek ilk 5 ülke
top5_lit = df.sort_values(by="Literacy (%)", ascending=False).head(5)[["Country", "Literacy (%)"]]

# 5. GDP per capita > 10.000 olan ülkeler
gdp_over_10k = df[df["GDP ($ per capita)"] > 10000][["Country", "GDP ($ per capita)"]]

# 6. Nüfus yoğunluğu en yüksek ilk 10 ülke
top10_density = df.sort_values(by="Pop. Density (per sq. mi.)", ascending=False).head(10)[["Country", "Pop. Density (per sq. mi.)"]]

# Sonuçları yazdır
print("1. Nüfusa göre azalan sırada:\n", pop_desc.head(), "\n")
print("2. GDP per capita’ya göre artan sırada:\n", gdp_asc.head(), "\n")
print("3. Nüfusu 10m+ olanlar:\n", pop_over_10m.head(), "\n")
print("4. En yüksek 5 okuryazarlık:\n", top5_lit, "\n")
print("5. GDP per capita > 10k:\n", gdp_over_10k.head(), "\n")
print("6. En yoğun 10 ülke:\n", top10_density, "\n")
