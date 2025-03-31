import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date

file_path = "datasets/owid-covid-data.csv"
df = pd.read_csv(file_path)

"""
TASK 1: Si richiede di verificare le dimensioni del dataset e i relativi metadati
"""
print("-" * 20 + "Task 1" + "-" * 20)
print(f"Il dataset è composto da: {df.shape[0]} righe e {df.shape[1]} colonne")

print(f"\n{df.columns}")
print("\n")
df.info()
print(f"\n{df.describe()}")
print(f"\n{df.head(5)}")

""" 
TASK 2: Si chiede di trovare per ogni continente:
a. il numero di casi fin dall'inizio della pandemia
b. la percentuale rispetto al totale mondiale del numero di casi
"""
print("-" * 20 + "Task 2" + "-" * 20)

# Controllo il funzionamento delle colonne new_cases e total_cases con un esempio
df["date"] = pd.to_datetime(df["date"], yearfirst=True)
filter_asia = df["continent"] == "Asia"
filter_dates = (df["date"].dt.year == 2020) & (df["date"].dt.month > 2)
print(df.loc[filter_asia & filter_dates, ["date", "new_cases", "total_cases"]].head(30))

# lista dei continenti
continents_list = np.sort(df.dropna(subset="continent")["continent"].unique())

# ciclo utilizzato per verifica dei valori di total_cases per le location aggregate
# for continent in continents_list:
#     continent_nations = df.loc[df["continent"] == continent, "location"].unique()
#     tot_continent_nations = df[df["location"].isin(continent_nations)].groupby("location")["total_cases"].last()
#     tot_continent = tot_continent_nations.sum()
#     print(f"{continent}: {tot_continent}")


tot_cases_per_location = df.groupby("location")["total_cases"].last()

continents_totals = tot_cases_per_location[tot_cases_per_location.index.isin(continents_list)]
print("\nTotale casi per ogni continente:")
print(continents_totals)

print("\nTotale casi per ogni continente (calcolo su new_cases)")
continents_totals_new_cases = df.groupby("continent")["new_cases"].sum()
print(f"{df.groupby("continent")["new_cases"].sum()}")

print("\nRapporto percentuale tra il risultato ottenuto considerando total_cases e new_cases")
percentage_diff = round(continents_totals / continents_totals_new_cases * 100,2)
print(percentage_diff)

# c'è una leggera differenza, pressocché trascurabile, tra il calcolo effettuato utilizzando i nuovi casi
# rispetto al calcolo effettuato considerando i casi totali, probabilmente dovuto alla presenza di alcuni
# valori nulli all'interno della colonna new_cases dovuti ad errori di registrazione dei dati,
# come riportato sul repository del documento
# In rare cases where our source reports a negative daily change due to a data correction, we set this metric to NA.


world_total = tot_cases_per_location["World"]
print(f"\nTotale casi mondiale: {world_total}")

print("\nPercentuale casi rispetto al totale mondiale per ogni continente:")
print(round(continents_totals / world_total * 100, 2))

""" 
TASK 3: Selezionare i dati relativi all'Italia nel 2022 e, poiché i nuovi casi vengono
registrati settimanalmente, filtrare via i giorni che non hanno misurazioni;
quindi mostrare con dei grafici adeguati: 
a. l'evoluzione dei casi totali dall'inizio alla fine dell'anno
b. il numero di nuovi casi rispetto alla data
"""
print("\n")
print("-" * 20 + "Task 3" + "-" * 20)

filter_italy = df["location"] == "Italy"
filter_2022 = df["date"].dt.year == 2022
df_italy_2022 = df[filter_italy & filter_2022]

df_italy_2022 = df_italy_2022[df_italy_2022["new_cases"] != 0]
print(f"\n{df_italy_2022.loc[:, ["date", "new_cases", "total_cases"]]}")

plt.plot(df_italy_2022["date"], df_italy_2022["total_cases"])

plt.title("Evoluzione dei casi totali 2022")
plt.xlabel("Date")
plt.ylabel("Casi totali")

plt.show()

df_italy_2022 = df_italy_2022.set_index(df_italy_2022["date"].dt.date)
df_italy_2022["new_cases"].plot(kind="bar")

plt.title("Nuovi casi 2022")
plt.xlabel("Date")
plt.ylabel("Nuovi casi")

plt.show()

"""
TASK 4: Riguardo le nazioni di Italia, Germania e Francia:
a. mostrare in un boxplot la differenza tra queste nazioni riguardo il numero di
pazienti in terapia intensiva (Intesive Care Unit, ICU, considerare quindi la colonna
icu_patients) da maggio 2022 (incluso) ad aprile 2023 (incluso)
b. scrivere un breve commento (una o due righe) riguardo che conclusioni possiamo
trarre osservando il grafico risultante
"""
print("\n")
print("-" * 20 + "Task 4" + "-" * 20)

filter_nations = df["location"].isin(["Italy", "Germany", "France"])
df_nations = df[filter_nations]

filter_dates = (df_nations["date"].dt.date >= date(2022, 5, 1)) & (df_nations["date"].dt.date <= date(2023, 4, 30))
df_nations = df_nations[filter_dates]

print(df_nations.groupby("iso_code")["icu_patients"].agg(["min", "mean", "median", "max"]))

sns.boxplot(data=df_nations, x="iso_code", y="icu_patients")

plt.title("BoxPlot ICU Patients")
plt.xlabel("Nations")
plt.ylabel("ICU Patients")
plt.show()

# Durante il periodo da Maggio 2022 ad Aprile 2023 il numero di pazienti mediamente presenti in ICU in Francia e
# Germania è stato paragonabile tra loro e molto più alto rispetto all'Italia. La Germania in particolare è il
# paese tra i 3 che ha raggiunto il picco più alto di pazienti in ICU ed ha avuto anche il valore mediano più alto
# durante il periodo considerato. L'italia invece è stato il paese tra i 3 in analisi che ha raggiunto il picco più
# basso di pazienti in ICU.

"""
TASK 5: Rigurado le nazioni di Italia, Germania, Francia e Spagna in tutto il 2021:
a. mostrare, in maniera grafica oppure numerica, la somma dei pazienti ospitalizzati
per ognuna (colonna hosp_patients)
b. se ci sono dati nulli, con un breve commento scrivere se può essere possibile gestirli
tramite sostituzione o meno
"""
print("\n")
print("-" * 20 + "Task 5" + "-" * 20)

filter_nations = df["location"].isin(["Italy", "Germany", "France", "Spain"])
filter_year = df["date"].dt.year == 2021
df = df[filter_nations & filter_year]
print(f"\nAnteprima del dataframe:\n{df.loc[:, ["iso_code", "date", "hosp_patients"]].head(10)}")

print("\nAnalisi dei dati nulli:")
filtro_nulli = df["hosp_patients"].isna()
num_val_nulli = df["hosp_patients"].isna().sum()
num_val_tot = df.shape[0]
print(f"\nNumero valori nulli: {num_val_nulli}, {round(num_val_nulli / num_val_tot * 100, 2)}% del totale")
print(f"\n{df.loc[filtro_nulli, ["iso_code", "date", "hosp_patients"]]}")

# Risulta molto difficile sostituire i valori nulli, considerato che sono relativi a tutto il 2021 per la nazione
# Germania, siccome è inverosimile che un intera nazione non abbia avuto alcun paziente ospedalizzato per l'intero
# anno 2021, sembra evidente che non siano stati registrati valori in merito all'ospedalizzazione
# dei pazienti per la Germania nell'anno in analisi.

print("\nTotale pazienti ospitalizzati per ognuna delle nazioni:")
print(df.groupby("iso_code")["hosp_patients"].sum())

df.groupby("iso_code")["hosp_patients"].sum().sort_values().plot(kind="bar")

plt.title("Pazienti ospedalizzati 2021")
plt.xlabel("Nazioni")
plt.ylabel("Totale pazienti ospedalizzati")

plt.show()
