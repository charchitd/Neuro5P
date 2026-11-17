"""Resolves per-agent votes to one consensus outcome and classifies
inter-dimensional conflicts rather than silencing them."""

from __future__ import annotations

from neuro5p.schema import AgentVote, Conflict, ConflictType, VoteType

_CONFLICT_PAIRS: dict[frozenset[str], ConflictType] = {
    frozenset({"performance", "parity"}): ConflictType.TYPE_A,
    frozenset({"proxies", "participation"}): ConflictType.TYPE_B,
    frozenset({"process", "parity"}): ConflictType.TYPE_C,
}


def resolve_votes(votes: list[AgentVote]) -> str:
    """pass / flag / block / escalate, per the v0.1 decision table."""
    if detect_conflicts(votes):
        return "escalate"
    if any(v.vote is VoteType.VETO for v in votes):
        return "block"
    if any(v.vote is VoteType.FLAG for v in votes):
        return "flag"
    return "pass"


def detect_conflicts(votes: list[AgentVote]) -> list[Conflict]:
    """A VETO on one dimension alongside a PASS on a paired, tension-prone
    dimension is a classified conflict, not noise to be averaged away."""
    by_id = {v.agent_id: v for v in votes}
    conflicts: list[Conflict] = []
    for pair, ctype in _CONFLICT_PAIRS.items():
        a, b = sorted(pair)
        va, vb = by_id.get(a), by_id.get(b)
        if va is None or vb is None:
            continue
        if {va.vote, vb.vote} == {VoteType.VETO, VoteType.PASS}:
            vetoer = a if va.vote is VoteType.VETO else b
            passer = b if vetoer == a else a
            conflicts.append(
                Conflict(
                    type=ctype,
                    agents=(vetoer, passer),
                    description=f"{vetoer} vetoed while {passer} passed on the same turn.",
                )
            )
    return conflicts


def generate_metacognitive_summary(
    votes: list[AgentVote], consensus: str
) -> tuple[str, list[str]]:
    """A structured self-report: what the consensus layer decided and what
    it is still uncertain about. Not a claim of subjective experience."""
    loud = sorted(votes, key=lambda v: -v.deviation_score)
    unknowns: list[str] = []
    parts = [f"Consensus: {consensus}."]
    for vote in loud:
        if vote.vote is not VoteType.PASS:
            parts.append(f"{vote.agent_id.capitalize()} column: {vote.reasoning}")
        if vote.deviation_score < 0.6 and vote.vote is not VoteType.PASS:
            unknowns.append(
                f"Whether the {vote.agent_id} signal reflects a genuine issue "
                f"or a borderline false positive (deviation={vote.deviation_score:.2f})."
            )
    if not unknowns:
        unknowns.append("None flagged above the borderline-confidence band.")
    return " ".join(parts), unknowns
