import pandas as pd

df = pd.read_csv('us_presidential_elections_1960_2024.csv', quotechar='"')
print(df[['election_year', 'candidate_name', 'party', 'won_election']].to_string(index=False))
