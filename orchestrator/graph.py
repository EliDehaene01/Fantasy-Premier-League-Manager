"""ARCHITECTURE.md 6c's StateGraph: (upstream of this graph - see the scope
note below) ingestion -> six specialist calls (genuinely parallel) -> the
bounded reaction round -> Manager (aggregate, call solver, retry-on-
infeasible, pick captain, narrate) -> a mode-dependent branch.

**Scope note on "ingestion" as a graph step**: ARCHITECTURE.md 8 also lists
`ingestion` as its own Kubernetes Job, separate from the `orchestrator`
Deployment. Re-running the full bronze/silver/gold pipeline synchronously
inside a graph node would duplicate that Job and reopen the no-lookahead
data-slicing logic ingestion/ already owns and tests (CLAUDE.md's first
hard constraint). This graph starts from an already-built ``player_pool``
in the initial state - the ingestion Job's output for live mode, or the
backtest engine's own no-lookahead slice for backtest mode - rather than
re-ingesting. ARCHITECTURE.md 6c's "ingestion" as the first step reads as
"the graph's run begins once fresh data exists," not literally a node
inside this StateGraph.

**Scope note on the infeasibility conditional edge**: ARCHITECTURE.md 6a's
retry-once-then-surface logic already happens INSIDE a single Manager call
(manager/service.py -> solver/optimizer.py's own two-attempt solve - see
that module's docstring). A graph-level loop back into the Manager node
with identical inputs would just reproduce the identical infeasible result
- there's nothing new for a second graph-level attempt to try. So the
conditional edge after Manager here is feasible -> mode branch, infeasible
-> declined/END, not a retry loop.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from manager.schemas import ManageRequest, ManagerPlayerFact
from shared.contracts import AgentArgument

from .callers import call_manage, call_react, call_specialist
from .state import GraphState

SPECIALIST_AGENTS = ("stats", "fixtures", "news", "contrarian", "template", "chips")
ADJUSTMENT_AGENTS = ("fixtures", "contrarian", "template", "news")


def build_graph(agent_caller=call_specialist, manage_caller=call_manage, react_caller=call_react, checkpointer=None):
    graph = StateGraph(GraphState)

    for agent_name in SPECIALIST_AGENTS:
        graph.add_node(agent_name, _make_specialist_node(agent_name, agent_caller))
        graph.add_edge(START, agent_name)
        graph.add_edge(agent_name, "reaction_round")

    graph.add_node("reaction_round", _make_reaction_node(react_caller))
    graph.add_edge("reaction_round", "manager")

    graph.add_node("manager", _make_manager_node(manage_caller))
    graph.add_conditional_edges("manager", _route_after_manager, {"proceed": "mode_branch", "declined": "declined"})

    graph.add_node("declined", _declined_node)
    graph.add_edge("declined", END)

    graph.add_node("mode_branch", _mode_branch_node)
    graph.add_edge("mode_branch", END)

    return graph.compile(checkpointer=checkpointer)


def _make_specialist_node(agent_name: str, agent_caller):
    def node(state: GraphState) -> dict:
        arg = agent_caller(agent_name, state["gameweek"], state["player_pool"])
        return {"first_round": {agent_name: arg.model_dump(mode="json")}}

    return node


def _make_reaction_node(react_caller):
    def node(state: GraphState) -> dict:
        first_round = {name: AgentArgument.model_validate(raw) for name, raw in state["first_round"].items()}
        return {"reactions": react_caller(state["gameweek"], first_round)}

    return node


def _effective_chip(state: GraphState) -> str | None:
    """The active chip for this gameweek is driven by the Chips agent's own
    recommendation (ARCHITECTURE.md 6a: "a Wildcard or Free Hit week (driven
    by the Chips agent's recommendation)"), not an independently pre-set
    state value. ``state.get("chip")`` is only a fallback for when the Chips
    agent's output isn't in ``first_round`` at all (e.g. an ablation run
    with Chips removed) - "missing opinions default to zero adjustment"'s
    spirit applied here too: no chip, not a guess.
    """
    chips_raw = state["first_round"].get("chips")
    if chips_raw is None:
        return state.get("chip")
    chips_arg = AgentArgument.model_validate(chips_raw)
    if chips_arg.chip_recommendation is None or chips_arg.chip_recommendation.chip is None:
        return None
    return chips_arg.chip_recommendation.chip.value


def _make_manager_node(manage_caller):
    def node(state: GraphState) -> dict:
        stats_raw = state["first_round"].get("stats")
        if stats_raw is None:
            return {"manage_result": {"feasible": False, "message": "stats agent produced no output", "squad": [], "narration": ""}}

        stats = AgentArgument.model_validate(stats_raw)
        adjustments = {
            name: AgentArgument.model_validate(state["first_round"][name])
            for name in ADJUSTMENT_AGENTS
            if name in state["first_round"]
        }
        effective_chip = _effective_chip(state)
        request = ManageRequest(
            gameweek=state["gameweek"],
            budget=state.get("budget", 100.0),
            free_transfers=state.get("free_transfers", 1),
            hit_cost=state.get("hit_cost", 4.0),
            chip=effective_chip,
            player_pool=[ManagerPlayerFact.model_validate(p) for p in state["player_pool"]],
            stats=stats,
            adjustments=adjustments,
        )
        result = manage_caller(request)
        # Overwrite state["chip"] with what was actually used (derived from
        # the Chips agent, not whatever the caller pre-set) - ManageResult
        # itself doesn't carry this, and the backtest engine's scoring
        # needs to know which chip effect (if any) to apply.
        return {"manage_result": result.model_dump(mode="json"), "chip": effective_chip}

    return node


def _route_after_manager(state: GraphState) -> str:
    return "proceed" if state["manage_result"]["feasible"] else "declined"


def _declined_node(state: GraphState) -> dict:
    # Rejection/timeout (and, here, infeasibility) just logs as declined -
    # no auto-retry or negotiation loop (ARCHITECTURE.md 6c).
    return {"approval_status": "declined"}


def _mode_branch_node(state: GraphState) -> dict:
    if state["mode"] == "backtest":
        # A 38-gameweek backtest can't pause for approval 38 times
        # (ARCHITECTURE.md 6c) - auto-accept the Manager's proposal.
        return {"approval_status": "auto_accepted"}
    # Live mode pauses here via interrupt(), holding state until
    # Command(resume=...) continues from exactly this point.
    decision = interrupt({"proposal": state["manage_result"]})
    return {"approval_status": "approved" if decision == "approve" else "rejected"}
