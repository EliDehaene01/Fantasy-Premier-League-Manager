"""Exploratory data analysis for the Stats agent dataset.

Run:  python -m agents.stats.eda

Produces:
  * docs/eda_summary.md      - plain-language findings
  * docs/figures/*.png       - the supporting charts

The point of this step is to look at the data *before* modelling: what the
target looks like, how lopsided minutes are, how prices move, and how often
players seem to miss games (our stand-in for "injury frequency", since this
dataset has no explicit injury column). Everything the model does later
should be defensible in light of what we see here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in this environment - write files only
import matplotlib.pyplot as plt
import numpy as np

from .data import REPO_ROOT, load_gameweeks

DOCS = REPO_ROOT / "docs"
FIG = DOCS / "figures"
POSITIONS = ["GK", "DEF", "MID", "FWD"]


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    return f"figures/{name}"


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def run() -> None:
    df = load_gameweeks()

    # "form" in FPL terms = average points over recent games. We approximate
    # the season-long picture with each player's mean points per gameweek.
    player_form = (
        df.groupby(["name", "position"])["total_points"].mean().reset_index(name="avg_points")
    )

    lines: list[str] = []
    add = lines.append

    add("# EDA summary - Stats agent dataset\n")
    add("Source: `merged_gws_2025-26.csv` (vaastav/Fantasy-Premier-League), "
        "one row per player per fixture.\n")
    add(f"After collapsing double gameweeks: **{len(df):,} player-gameweek rows**, "
        f"**{df['name'].nunique():,} players**, gameweeks {df.GW.min()}-{df.GW.max()}.\n")

    # ------------------------------------------------------------------
    # 1. The target: total_points
    # ------------------------------------------------------------------
    tp = df["total_points"]
    played = df[df["minutes"] > 0]["total_points"]
    add("## 1. The target (`total_points`) is small, skewed and zero-heavy\n")
    add(f"- Mean **{_fmt(tp.mean())}** points per player-gameweek, median **{tp.median():.0f}**, "
        f"max **{tp.max():.0f}**.")
    add(f"- **{(tp == 0).mean():.0%} of rows are exactly 0** - mostly players who did not play.")
    add(f"- Among rows where the player actually got minutes, the mean is "
        f"**{_fmt(played.mean())}** (still skewed - a few hauls pull the tail).")
    add(f"- Points can be **negative** (min **{tp.min():.0f}**: yellow/red cards, own goals).\n")
    add("*Why it matters:* this is count-like, non-negative-ish, heavily zero-inflated "
        "data. A plain linear regression assuming a symmetric bell curve is a poor fit; "
        "a Poisson model (log link, variance grows with the mean) is closer to right, and "
        "tree models don't care about the shape at all. It also means **RMSE will be "
        "dominated by the rare big hauls** while MAE reflects the typical row - we report "
        "both.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tp, bins=range(int(tp.min()), int(tp.max()) + 2), color="#3b7dd8", edgecolor="white")
    ax.set_title("Distribution of total_points per player-gameweek")
    ax.set_xlabel("total_points")
    ax.set_ylabel("rows (log scale)")
    ax.set_yscale("log")
    add(f"![target distribution]({_save(fig, 'target_distribution.png')})\n")

    # ------------------------------------------------------------------
    # 2. Form by position
    # ------------------------------------------------------------------
    add("## 2. Form (average points) differs by position\n")
    add("| position | players | mean avg-points | median | 90th pct |")
    add("|---|---|---|---|---|")
    for pos in POSITIONS:
        s = player_form[player_form.position == pos]["avg_points"]
        add(f"| {pos} | {len(s)} | {_fmt(s.mean())} | {_fmt(s.median())} | {_fmt(s.quantile(0.9))} |")
    add("")
    add("*Reading it:* midfielders and forwards have the highest ceilings (attacking "
        "returns + the extra goal points for MID), goalkeepers are the most predictable "
        "(tight spread), defenders sit in between and are effectively two populations - a "
        "nailed-on starter in a good defence vs. a rotated one in a bad defence.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [player_form[player_form.position == p]["avg_points"].values for p in POSITIONS]
    ax.boxplot(data, tick_labels=POSITIONS, showfliers=False)
    ax.set_title("Season average points per gameweek, by position")
    ax.set_ylabel("avg points / GW")
    add(f"![form by position]({_save(fig, 'form_by_position.png')})\n")

    # ------------------------------------------------------------------
    # 3. Minutes
    # ------------------------------------------------------------------
    mins = df["minutes"]
    add("## 3. Minutes are bimodal - you either play ~90 or you don't play\n")
    add(f"- **{(mins == 0).mean():.0%}** of rows are 0 minutes.")
    add(f"- **{(mins >= 60).mean():.0%}** of rows are 60+ minutes (the threshold for the "
        f"2-point appearance bonus).")
    add(f"- Only **{((mins > 0) & (mins < 60)).mean():.0%}** of rows are 'cameo' minutes "
        f"(1-59).\n")
    add("*Why it matters:* predicting points is largely predicting *minutes* first. The "
        "middle ground barely exists, so a feature like recent start-rate should carry a "
        "lot of weight, and per-90 normalisation is needed so a good sub isn't buried by "
        "his low minute count.\n")
    add("| position | % rows 0 min | % rows 60+ min | mean minutes (all rows) |")
    add("|---|---|---|---|")
    for pos in POSITIONS:
        g = df[df.position == pos]["minutes"]
        add(f"| {pos} | {(g == 0).mean():.0%} | {(g >= 60).mean():.0%} | {_fmt(g.mean(), 1)} |")
    add("")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mins, bins=range(0, 100, 5), color="#4a9c6d", edgecolor="white")
    ax.set_title("Distribution of minutes per player-gameweek")
    ax.set_xlabel("minutes")
    ax.set_ylabel("rows")
    add(f"![minutes distribution]({_save(fig, 'minutes_distribution.png')})\n")

    # ------------------------------------------------------------------
    # 4. Price changes
    # ------------------------------------------------------------------
    d = df.sort_values(["name", "GW"]).copy()
    d["price_change"] = d.groupby("name")["price"].diff()
    pc = d["price_change"].dropna()
    nonzero = pc != 0
    add("## 4. Price changes are tiny, rare and mostly Ā£0.1 steps\n")
    add(f"- **{(pc == 0).mean():.0%}** of gameweek-to-gameweek price moves are exactly Ā£0.0.")
    add(f"- When a price does move it is almost always **around Ā±Ā£0.1** "
        f"({(pc.abs().between(0.05, 0.15)).sum() / nonzero.sum():.0%} of non-zero moves).")
    add(f"- Season price range across players: Ā£{df.price.min():.1f}m - Ā£{df.price.max():.1f}m.")
    add(f"- Biggest single-GW rise **+Ā£{pc.max():.1f}m**, biggest fall **Ā£{pc.min():.1f}m**.\n")
    add("*Why it matters:* raw price change per gameweek is a weak, sparse signal. It is "
        "more useful as **momentum** - the cumulative move over the last 3-5 gameweeks and "
        "its direction - which reflects what the crowd of managers thinks about a player's "
        "near-term prospects. That is how the price feature is engineered.\n")
    add("| position | mean season price | % GWs with a price move |")
    add("|---|---|---|")
    for pos in POSITIONS:
        g = d[d.position == pos]
        moves = g["price_change"].dropna()
        add(f"| {pos} | Ā£{g.price.mean():.1f}m | {(moves != 0).mean():.0%} |")
    add("")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pc[nonzero], bins=np.arange(-0.65, 0.7, 0.1), color="#c67b2e", edgecolor="white")
    ax.set_title("Non-zero gameweek-to-gameweek price changes")
    ax.set_xlabel("Ā£m change")
    ax.set_ylabel("rows")
    add(f"![price changes]({_save(fig, 'price_changes.png')})\n")

    # ------------------------------------------------------------------
    # 5. "Injury frequency" (proxy)
    # ------------------------------------------------------------------
    add("## 5. Injury / unavailability frequency (proxy)\n")
    add("This dataset has **no injury column**. In the live system the News agent "
        "handles availability via team-news RAG. For EDA we use a behavioural proxy:\n")
    add("> A player is counted as **unavailable in gameweek N** if they played 60+ min in "
        "*both* gameweeks N-2 and N-1, then played **0 minutes** in N. That pattern - an "
        "established starter suddenly absent - is usually injury or suspension rather than "
        "a tactical call.\n")

    d["m1"] = d.groupby("name")["minutes"].shift(1)
    d["m2"] = d.groupby("name")["minutes"].shift(2)
    d["was_nailed"] = (d["m1"] >= 60) & (d["m2"] >= 60)
    d["dropped_out"] = d["was_nailed"] & (d["minutes"] == 0)
    overall = d.loc[d["was_nailed"], "dropped_out"].mean()
    add(f"Across all established starters, **{overall:.1%}** of gameweeks see them drop out "
        f"like this - so a nailed player has roughly a **1-in-{round(1 / overall)}** chance "
        f"of missing any given gameweek.\n")
    add("| position | established-starter GWs | unavailable next GW | rate |")
    add("|---|---|---|---|")
    rows_by_pos = []
    for pos in POSITIONS:
        g = d[(d.position == pos) & d["was_nailed"]]
        rate = g["dropped_out"].mean() if len(g) else float("nan")
        rows_by_pos.append((pos, len(g), int(g["dropped_out"].sum()), rate))
        add(f"| {pos} | {len(g):,} | {int(g['dropped_out'].sum()):,} | {rate:.1%} |")
    add("")
    worst = max(rows_by_pos, key=lambda r: (r[3] if r[3] == r[3] else -1))
    add(f"*Reading it:* **{worst[0]}s** drop out most often ({worst[3]:.1%}), consistent "
        "with forwards and centre-backs carrying more knocks. The practical takeaway for "
        "the model: recent minutes and start-rate are not just 'form', they are also an "
        "**availability estimate** - and the Manager still gets a hard veto from the "
        "News agent on top of whatever this model predicts.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r[0] for r in rows_by_pos], [r[3] * 100 for r in rows_by_pos], color="#b34b4b")
    ax.set_title("Proxy unavailability rate for established starters, by position")
    ax.set_ylabel("% of gameweeks dropped out")
    add(f"![injury proxy]({_save(fig, 'injury_proxy_by_position.png')})\n")

    # ------------------------------------------------------------------
    # 6. Quick correlation check
    # ------------------------------------------------------------------
    add("## 6. What already correlates with next-gameweek points?\n")
    d2 = df.sort_values(["name", "GW"]).copy()
    d2["next_points"] = d2.groupby("name")["total_points"].shift(-1)
    cand = {
        "last GW points": d2["total_points"],
        "last GW minutes": d2["minutes"],
        "last GW ict_index": d2["ict_index"],
        "last GW xGI": d2["expected_goal_involvements"],
        "last GW bps": d2["bps"],
        "price": d2["price"],
    }
    corrs = {k: v.corr(d2["next_points"]) for k, v in cand.items()}
    add("Pearson correlation of a single past-gameweek stat with the *following* "
        "gameweek's points (rough guide, not the model):\n")
    add("| stat (gameweek N) | corr with points in N+1 |")
    add("|---|---|")
    for k, v in sorted(corrs.items(), key=lambda kv: -kv[1]):
        add(f"| {k} | {_fmt(v)} |")
    add("")
    add("*Reading it:* recent minutes (~0.5) and ICT (~0.45) carry the most single-signal "
        "information; last gameweek's points on its own is ~0.4. Nothing reaches 0.6. This "
        "is why the model needs many weak features combined, and why beating a naive "
        "rolling-average baseline by a wide margin is unlikely - points are genuinely "
        "noisy week to week.\n")

    add("## Headline takeaways for modelling\n")
    add("1. **Predict minutes before points.** ~61% of rows are zeros; recent minutes and "
        "start-rate are the highest-value features.")
    add("2. **Use MAE as the primary metric, RMSE as secondary.** RMSE is hostage to a "
        "handful of 15+ point hauls.")
    add("3. **Poisson over plain linear.** The target is skewed, zero-inflated and "
        "(mostly) non-negative count data.")
    add("4. **Expect a modest beat over the naive baseline.** The best single stat "
        "(recent minutes) correlates ~0.5 with next-GW points; last-GW points alone ~0.4.")
    add("5. **Availability is a feature, not just an agent concern** - but the Injuries "
        "agent still holds the hard veto.\n")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS / 'eda_summary.md'}")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    run()
