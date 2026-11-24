# Tap Scanner

Scans US stocks and Coinbase crypto for Wyckoff second and third taps every 30 minutes,
sends new third taps to your phone, and shows everything on a dashboard with the range and
every tap drawn on the chart.

- **Dashboard:** `https://<your-github-name>.github.io/<this-repo>/`
- **Settings:** `config.json` (watchlists, timeframes, rules, which alerts you get)
- **Rules the scanner follows:** [RULES.md](RULES.md)

Everything runs on GitHub's free tier. The repository needs to be **public** for the free
dashboard and unlimited scan minutes, so anyone with the link can see your signals. Your API
keys and ntfy topic stay private in the repository's secrets.

## One-time setup

### 1. Secrets (your keys)
Repository **Settings → Secrets and variables → Actions → New repository secret**, three times:

| Name | Value |
|---|---|
| `APCA_API_KEY_ID` | Alpaca paper-trading **API Key ID** (the shorter one, usually starts with `PK`) |
| `APCA_API_SECRET_KEY` | Alpaca paper-trading **Secret Key** |
| `NTFY_TOPIC` | Your ntfy topic name, the same one you subscribed to in the ntfy app |

### 2. Turn on the dashboard
**Settings → Pages**. Under *Build and deployment*, set Source to **Deploy from a branch**,
branch **main**, folder **/docs**, then **Save**. After a minute or two the page shows the
dashboard's address.

### 3. Run the first scan
**Actions** tab. If GitHub asks, click **I understand my workflows, go ahead and enable them**.
Open **Scan** on the left, then **Run workflow → Run workflow**. The first run downloads
history, so it takes a few minutes. After that it runs by itself every 30 minutes.

The first scan doesn't send phone alerts for setups that already exist, so you don't get a
flood. From then on you get alerts for new ones.

## Day to day

- **Scan now:** the dashboard button opens the Run workflow page.
- **Change settings:** on the dashboard's settings page. The first time, connect the browser
  with a GitHub fine-grained token (the page shows the steps): *Only select repositories* →
  this repository, with **Contents: Read and write** and **Actions: Read and write**. The token
  stays in that browser only. Then add or remove tickers and coins, switch timeframes and
  rules, and press **Save** or **Save & scan now**. You can also edit `config.json` on GitHub.
- **Stocks** are scanned on weekdays from 9:00 to 17:00 New York time. Outside those hours
  the dashboard keeps the last results. **Crypto** is scanned around the clock.
- **Pause everything:** Actions → Scan → **⋯ → Disable workflow**.

## Settings reference (`config.json`)

| Setting | Values |
|---|---|
| `stocks.universe` | `"my list"` (the `tickers`), `"most active 100"`, `"all US stocks"` |
| `crypto.universe` | `"my list"` (the `products`), `"top 25"`, `"top 50"`, `"top 100"`, `"top 200"`, `"all USD pairs"` |
| `*.timeframes` | any of `"5m"`, `"15m"`, `"30m"`, `"1h"`, `"4h"`, `"1D"`, `"1W"` |
| `rules.tap2_return` | `"either"`, `"quarter"`, `"two_closes"`, `"close"` |
| `rules.require_ar_break_of_structure` | `true` / `false` |
| `rules.allow_nested_ranges` | `true` / `false` |
| `rules.max_opposite_deviations` | how many separate deviations of the other end of the range are allowed before tap 3 (default 1 = red flag; one more ignores the setup; 0 = must stay fresh) |
| `rules.trend_lookback` | tap 1 must close below (selling climax) or above (buying climax) every close of this many previous candles, so ranges follow the trend; `0` turns it off |
| `alerts.tap3_potential`, `alerts.tap3_confirmed`, `alerts.model1_forming` | `true` / `false` |
| `alerts.max_bars_ago` | only alert for taps this many candles old or newer |
| `display.timezone` | e.g. `"America/Los_Angeles"` |
| `display.history_days` | how far back the dashboard's *all setups* view goes |

## Running it yourself (optional)
```
pip install -r requirements.txt pytest
python -m pytest tests                 # tests on synthetic setups
APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... NTFY_TOPIC=... SCAN_MARKETS=stocks,crypto python scan.py
python -m http.server -d docs          # dashboard at http://localhost:8000
```
