import { useEffect, useState } from "react";
import SquadView from "./components/SquadView.jsx";
import Transcript from "./components/Transcript.jsx";

// CLAUDE.md hard constraint: the frontend never talks to Postgres or any
// backend, directly or indirectly - these two fetches (the index, then one
// gameweek file) are the ONLY network calls this app ever makes, and both
// are plain static files under public/data/, committed by frontend-export
// (see that package's own README for how they get there).
const DATA_BASE = `${import.meta.env.BASE_URL}data`;

export default function App() {
  const [gameweeks, setGameweeks] = useState([]);
  const [selected, setSelected] = useState(null);
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${DATA_BASE}/index.json`)
      .then((r) => r.json())
      .then((data) => {
        const gws = data.gameweeks.sort((a, b) => b.gameweek - a.gameweek);
        setGameweeks(gws);
        if (gws.length > 0) setSelected(gws[0].gameweek);
      })
      .catch((e) => setError(`Couldn't load the gameweek index: ${e.message}`));
  }, []);

  useEffect(() => {
    if (selected == null) return;
    setRecord(null);
    fetch(`${DATA_BASE}/gw${selected}.json`)
      .then((r) => r.json())
      .then(setRecord)
      .catch((e) => setError(`Couldn't load gameweek ${selected}: ${e.message}`));
  }, [selected]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>FPL Agents</h1>
        <p className="subtitle">A multi-agent FPL manager's weekly recommendations, in full.</p>
      </header>

      {error && <p className="error">{error}</p>}

      {gameweeks.length > 0 && (
        <nav className="gw-picker">
          {gameweeks.map((gw) => (
            <button
              key={gw.gameweek}
              className={gw.gameweek === selected ? "gw-btn active" : "gw-btn"}
              onClick={() => setSelected(gw.gameweek)}
            >
              GW{gw.gameweek}
              <span className={`badge badge-${gw.state}`}>{gw.state}</span>
            </button>
          ))}
        </nav>
      )}

      {record && (
        <main className="record">
          <SquadView record={record} />
          <Transcript transcript={record.transcript} />
        </main>
      )}
    </div>
  );
}
