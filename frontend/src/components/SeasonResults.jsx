// ARCHITECTURE.md 9 step 3 / TODO.md Phase 8: a small report view for the
// backtest's season totals and ablations - real numbers from
// backtest/run_full_season.py's actual run, not illustrative placeholders.

const AGENT_LABELS = {
  fixtures: "Fixtures",
  contrarian: "Contrarian",
  template: "Template",
  chips: "Chips",
  news: "News",
};

export default function SeasonResults({ data }) {
  if (!data) return null;
  const gws = Object.entries(data.net_points_by_gw).sort((a, b) => Number(a[0]) - Number(b[0]));
  const maxPts = Math.max(...gws.map(([, p]) => p));

  return (
    <section className="season-results">
      <h2>Season backtest: {data.season}</h2>
      <p className="subtitle">
        Gameweeks {data.start_gw}–{data.end_gw}, walked forward with no lookahead - real trained model, real solver, real archive data.
      </p>

      <div className="benchmark-row">
        <div className="benchmark">
          <span className="benchmark-value">{data.baseline_total_points}</span>
          <span className="benchmark-label">This system</span>
        </div>
        <div className="benchmark">
          <span className="benchmark-value">{data.never_transfer_total_points}</span>
          <span className="benchmark-label">Never-transfer baseline</span>
        </div>
        <div className="benchmark benchmark-unavailable">
          <span className="benchmark-value">—</span>
          <span className="benchmark-label" title={data.average_manager_unavailable_reason}>
            FPL average manager (unavailable)
          </span>
        </div>
      </div>
      <p className="benchmark-delta">
        {(data.baseline_total_points - data.never_transfer_total_points).toFixed(0)} points ahead of the never-transfer baseline over the season.
      </p>

      <h3>Points by gameweek</h3>
      <div className="gw-chart">
        {gws.map(([gw, pts]) => (
          <div key={gw} className="gw-bar-wrap" title={`GW${gw}: ${pts} pts`}>
            <div className="gw-bar" style={{ height: `${(pts / maxPts) * 100}%` }} />
          </div>
        ))}
      </div>

      <h3>Per-agent ablations</h3>
      <p className="subtitle">Season rerun with one specialist removed at a time - the delta is what that agent was actually worth.</p>
      <ul className="ablation-list">
        {Object.entries(data.ablation_deltas).map(([agent, delta]) => (
          <li key={agent} className={delta >= 0 ? "ablation-positive" : "ablation-negative"}>
            <span>{AGENT_LABELS[agent] ?? agent}</span>
            <span>{delta >= 0 ? "+" : ""}{delta.toFixed(0)} pts</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
