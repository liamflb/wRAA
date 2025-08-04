import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from unidecode import unidecode

data = pd.read_csv("full_stats.csv")
razz_csv = pd.read_csv("mlb_ids.csv")

mlb_ids_plus_name = razz_csv[['MLBAMID', 'Name']]
name_to_id = mlb_ids_plus_name.set_index('Name')['MLBAMID'].to_dict()

def player_lookup(name, start_year, end_year = None):
    if name not in name_to_id:
        raise KeyError("Player ID not found")
    player_id = name_to_id[name]

    if end_year == None:
        years = [start_year]
    else:
        years = list(range(start_year, end_year + 1))

    mask = (data['player_id'] == player_id) & (data['year'].isin(years))
    player_data = data.loc[mask]

    player_woba_series = player_data['liam_xwoba']

    plate_apps = len(player_woba_series)

    player_woba = player_woba_series.mean()

    player_woba = np.round(player_woba, decimals=3)

    return player_woba


#-----------Do xwoba by player to get better graph---------------------#

mlb_predictions =  pd.read_csv("expected_stats.csv")
mlb_woba = mlb_predictions['est_woba']
mlb_real_woba = mlb_predictions['woba']

def name_helper(first_last):
    lst = first_last.split(', ')

    name = f"{lst[1]} {lst[0]}"

    clean_name = unidecode(name)

    return clean_name

mlb_predictions['qualified_batters'] = mlb_predictions['last_name, first_name'].apply(name_helper)

qualified_hitters = mlb_predictions['qualified_batters']

name_to_mlb_woba = mlb_predictions.set_index('qualified_batters')['est_woba'].to_dict()

my_wobas = []
mlb_wobas = []
names = []

for name in qualified_hitters:
    try:
        # attempt to look up and append
        my_wobas.append(player_lookup(name, 2024))
        mlb_wobas.append(name_to_mlb_woba[name])
        names.append(name)
    except KeyError:
        # whenever player_lookup or the dict lookup fails
        #print(f"Player ID not found for {name!r}")
        continue

# cleaned_wobas = [woba for woba in my_wobas if ~np.isnan(woba[0])]
# just_wobas = [woba[0] for woba in cleaned_wobas]
# print(just_wobas)
# league_woba = np.mean(just_wobas)
# print(league_woba)

x = np.linspace(0.24, 0.5, 5)
y = x

my_wobas = np.array(my_wobas)
mlb_wobas = np.array(mlb_wobas)
names = np.array(names)

mask = ~np.isnan(my_wobas)
my_wobas_clean = my_wobas[mask]
mlb_wobas_clean = mlb_wobas[mask]
names_clean = names[mask]

slope, intercept, r_value, p_value, std_err = stats.linregress(my_wobas_clean, mlb_wobas_clean)

r_squared = r_value ** 2

print(f"R^2 Value: {r_squared}")

r_squared = np.round(r_squared, decimals=2)

plt.scatter(my_wobas_clean, mlb_wobas_clean, c='purple', marker='o',alpha=0.5 )
plt.plot(x,y, c='blue')
plt.xlabel("Liam's xwOBA")
plt.annotate(f"R^2 Value: {r_squared}", xy=[0.05, 0.95], xycoords='axes fraction')
plt.ylabel("MLB xwOBA")
plt.title("My xwOBA vs MLB's xwOBA")
plt.show()

liam_wobas_by_player = pd.DataFrame({
    'player': names_clean,
    'liam_xwOBA': my_wobas_clean,
})

liam_wobas_by_player = liam_wobas_by_player.sort_values(by="liam_xwOBA", ascending=False)
liam_wobas_by_player.to_csv("player_wobas.csv", index=False)

all = data['liam_xwoba'][data['year'] == 2024]

print(np.mean(all))