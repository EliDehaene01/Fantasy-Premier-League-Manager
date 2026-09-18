# Frontend

React + Vite, deployed to GitHub Pages. Reads only static JSON under
`public/data/` (written by `../frontend-export/`) - CLAUDE.md hard
constraint: **never** calls Postgres or any backend, directly or
indirectly. Verified by inspection, not just assumption: `src/App.jsx` is
the only place this app calls `fetch()`, and both calls target
`${BASE_URL}data/*.json` - grep the rest of `src/` for `fetch(` to confirm
if this changes.

## Develop

```bash
npm install
npm run dev       # dev server, reads public/data/ directly
npm run build     # writes dist/ - public/ is copied in at build time,
                   # so re-run this after frontend-export writes new JSON
npm run preview   # serves dist/ (not public/) - rebuild first to see changes
```

## Data contract

One JSON file per gameweek (`public/data/gw<N>.json`) plus an
`index.json` listing what's available - see
`../frontend-export/export_gameweek.py`'s module docstring for the full
shape (pending vs. final state, squad, transcript, points).

## Deploying to GitHub Pages

Not done as part of building this app - pushing to a public GitHub Pages
site is a one-way, publicly-visible action, left for an explicit ask
rather than assumed. `vite.config.js`'s `base` is already set to this
repo's name, ready for `npm run build` + publishing `dist/` to a
`gh-pages` branch (or GitHub Actions) whenever that's wanted.
