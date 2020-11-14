import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

team = "Cleveland Browns"
base_url = "https://www.pro-football-reference.com/years/2020/week_{}.htm"
max_week = 9

offensive_scores = []
defensive_scores = []
for week in range(1, max_week + 1):
    url = base_url.format(week)
    r = requests.get(url)
    soup = bs(r.text, "html.parser")
    games = soup.find_all(attrs={"class": "teams"})
    offensive_score = defensive_score = "BYE"
    for game in games:
        table_rows = game.find_all("td")
        for i, tr in enumerate(table_rows):
            if tr.string in [team]:
                offensive_team = tr.string
                offensive_score = int(table_rows[i + 1].string)
                try:
                    # Hacky
                    defensive_team = table_rows[i + 3].string
                    defensive_score = int(table_rows[i + 4].string)
                except IndexError:
                    defensive_team = table_rows[i - 3].string
                    defensive_score = int(table_rows[i - 2].string)
                offensive_scores.append(offensive_score)
                defensive_scores.append(defensive_score)
    if offensive_score == "BYE":
        offensive_scores.append(offensive_score)
        defensive_scores.append(defensive_score)

indexes = [f"{team}'s Points Scored"]
# columns = list(map(lambda x: f"Week {x}", range(1, max_week + 1)))
columns = list(range(1, max_week + 1))
values = [defensive_scores]
df = pd.DataFrame(index=indexes, columns=columns, data=values)
print(df.to_markdown(tablefmt="github"))
