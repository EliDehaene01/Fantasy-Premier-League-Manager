// Renders the chosen/proposed team, the pending-vs-final state, and (once
// played) actual points - ARCHITECTURE.md 8a's core per-gameweek view.

function playerLabel(p) {
  return p.name ? `${p.name} (${p.position ?? "?"}${p.team ? `, ${p.team}` : ""})` : `Player #${p.player_id}`;
}

function PlayerRow({ player, captainId, viceCaptainId, isFinal }) {
  const isCaptain = player.player_id === captainId;
  const isVice = player.player_id === viceCaptainId;
  return (
    <li className={player.starting ? "player starting" : "player bench"}>
      <span className="player-name">
        {playerLabel(player)}
        {isCaptain && <span className="armband captain"> (C)</span>}
        {isVice && <span className="armband vice"> (VC)</span>}
      </span>
      {isFinal && (
        <span className="player-points">
          {player.actual_points ?? "–"} pts
          {isCaptain && player.starting && " ×2"}
        </span>
      )}
    </li>
  );
}

export default function SquadView({ record }) {
  const isFinal = record.state === "final";
  const starting = record.squad.filter((p) => p.starting);
  const bench = record.squad.filter((p) => !p.starting);

  return (
    <section className="squad-view">
      <div className="gw-summary">
        <h2>
          Gameweek {record.gameweek}
          <span className={`badge badge-${record.state}`}>{record.state}</span>
        </h2>
        {record.chip && <p className="chip-used">Chip played: {record.chip.replace("_", " ")}</p>}
        <p>
          {record.transfers_made ?? 0} transfer(s), {record.hits_taken ?? 0} hit(s)
          {record.hit_points_cost ? ` (−${record.hit_points_cost} pts)` : ""}
        </p>
        {isFinal && record.points && (
          <p className="points-total">
            <strong>{record.points.net} pts</strong> scored this gameweek
            {record.hit_points_cost ? ` (${record.points.gross} gross)` : ""}
          </p>
        )}
      </div>

      {record.narration && <p className="narration">{record.narration}</p>}

      <div className="squad-lists">
        <div>
          <h3>Starting XI</h3>
          <ul>
            {starting.map((p) => (
              <PlayerRow key={p.player_id} player={p} captainId={record.captain_id} viceCaptainId={record.vice_captain_id} isFinal={isFinal} />
            ))}
          </ul>
        </div>
        <div>
          <h3>Bench</h3>
          <ul>
            {bench.map((p) => (
              <PlayerRow key={p.player_id} player={p} captainId={record.captain_id} viceCaptainId={record.vice_captain_id} isFinal={isFinal} />
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
