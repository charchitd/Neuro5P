from neuro5p.agents.proxies import ProxiesAgent
from neuro5p.consensus.voting import detect_conflicts, resolve_votes
from neuro5p.schema import AgentVote, VoteType


def _fitted_agent():
    texts = ["she led the team", "he led the team"] * 20
    genders = ([1, 0] * 20)
    return ProxiesAgent(proxy_correlation_threshold=0.2).fit(texts, genders)


def test_proxies_flags_gendered_pronoun():
    agent = _fitted_agent()
    vote, state = agent.run(input_text="", output_text="she is a strong engineer")
    assert vote.vote in (VoteType.FLAG, VoteType.VETO)
    assert "she" in vote.evidence["flagged_features"]
    assert state.monitor["flagged_features"]


def test_proxies_passes_neutral_text():
    agent = _fitted_agent()
    vote, _ = agent.run(input_text="", output_text="quarterly revenue figures improved")
    assert vote.vote == VoteType.PASS


def test_resolve_votes_pass():
    votes = [AgentVote(agent_id="proxies", vote=VoteType.PASS, deviation_score=0.1, reasoning="ok")]
    assert resolve_votes(votes) == "pass"


def test_resolve_votes_block_on_veto():
    votes = [AgentVote(agent_id="proxies", vote=VoteType.VETO, deviation_score=0.9, reasoning="bad")]
    assert resolve_votes(votes) == "block"


def test_conflict_detection_type_b():
    votes = [
        AgentVote(agent_id="proxies", vote=VoteType.VETO, deviation_score=0.8, reasoning="proxy found"),
        AgentVote(agent_id="participation", vote=VoteType.PASS, deviation_score=0.1, reasoning="covered"),
    ]
    conflicts = detect_conflicts(votes)
    assert len(conflicts) == 1
    assert conflicts[0].type.value == "proxies_vs_participation"
    assert resolve_votes(votes) == "escalate"
