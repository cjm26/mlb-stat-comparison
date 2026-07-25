# MLB Stat Comparator

Look up any MLB player and see how their current-season stats rank against the
rest of the league — percentile plus a histogram of the league distribution for
each stat.

- Stat type follows the player: hitters get batting, pitchers get pitching, and
  two-way players get both.
- Playing time (PA / IP) is shown alongside the rate stats, since it's the
  sample size everything else rests on.
- Rate and counting stats are compared against players with meaningful playing
  time (1 PA and 0.25 IP per team game, scaled as the season progresses) so that
  small samples don't distort the distribution. Playing time itself is compared
  against every player who filled that role.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Stats come from the MLB Stats API (`statsapi.mlb.com`), regular season only,
refreshed every 30 minutes. This is an undocumented public endpoint with no
stability guarantee.

Data and content are proprietary to MLB Advanced Media, L.P. This project is
not affiliated with or endorsed by MLB or MLBAM.
