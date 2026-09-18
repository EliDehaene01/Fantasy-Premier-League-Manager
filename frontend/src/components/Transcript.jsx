// The full six-agent debate transcript: each specialist's first-round
// argument plus its bounded, text-only reaction (ARCHITECTURE.md 8a).

const AGENT_LABELS = {
  stats: "Stats",
  fixtures: "Fixtures",
  news: "News",
  contrarian: "Contrarian",
  template: "Template",
  chips: "Chips",
};

function RecommendationList({ title, items, kind }) {
  if (!items || items.length === 0) return null;
  return (
    <div className={`rec-list rec-${kind}`}>
      <strong>{title}:</strong>
      <ul>
        {items.map((item, i) => (
          <li key={i}>
            {item.name ?? `Player #${item.player_id}`}
            {kind === "recommendation" && ` (conviction ${item.conviction?.toFixed(2)}${item.predicted_points != null ? `, ${item.predicted_points} pts predicted` : ""})`}
            {kind === "veto" && ` — ${item.status}${item.confidence != null ? ` (${Math.round(item.confidence * 100)}% confident)` : ""}`}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Transcript({ transcript }) {
  if (!transcript || transcript.length === 0) return null;
  return (
    <section className="transcript">
      <h3>Agent debate</h3>
      {transcript.map((entry) => (
        <article key={entry.agent} className="transcript-entry">
          <h4>{AGENT_LABELS[entry.agent] ?? entry.agent}</h4>
          <p className="reasoning">{entry.reasoning}</p>
          <RecommendationList title="Recommends" items={entry.recommendations} kind="recommendation" />
          <RecommendationList title="Vetoes" items={entry.vetoes} kind="veto" />
          {entry.chip_recommendation && entry.chip_recommendation.chip && (
            <p className="chip-rec">Suggests: {entry.chip_recommendation.chip.replace("_", " ")} — {entry.chip_recommendation.reasoning}</p>
          )}
          {entry.reaction && <p className="reaction">↳ {entry.reaction}</p>}
        </article>
      ))}
    </section>
  );
}
