import pandas as pd

full_stats = pd.read_csv('full_stats.csv')
league_woba = full_stats['liam_xwoba'].mean()

qualified_hitter_stats = pd.read_csv("player_wobas.csv")

woba_scale = 1.236

qualified_hitter_stats['xwRAA'] = (((qualified_hitter_stats['liam_xwOBA'] - league_woba) / woba_scale) * qualified_hitter_stats['pas']).round(2)
full_stats['xwRAA'] = (full_stats['liam_xwoba'] - league_woba) / woba_scale


qualified_hitter_stats.to_csv("player_xwRAA.csv", index=False)
