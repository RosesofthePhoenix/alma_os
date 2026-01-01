# alma_os

ALMA OS

A lightweight Dash multipage shell with a neon-dark theme, sidebar navigation, and placeholder pages for future expansion.

## Getting started

1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

If you want Spotify logging to reuse your cached token, ensure you launch from the project root (where `.cache` is written) or set:
```
export SPOTIPY_CACHE="/Users/raycraigantelo/Documents/alma_os/.cache"
```

## Structure

- `app.py`: Dash entrypoint with sidebar routing.
- `alma/ui`: Layout, theme, and page definitions.
- `profiles/default.json`: Placeholder profile config.
- `sessions/current/adaptive_mode.json`: Adaptive mode flag (defaults to OFF).
- `data/`: Data directory placeholder.
- `external/`: Existing scripts preserved.
