"""Abstract MAPE-K agent that every 5P column implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from neuro5p.schema import AgentVote, MAPEKState


class BaseAgent(ABC):
    """One cortical-column analogue: an independent monitor running its own
    Monitor -> Analyse -> Plan -> Execute loop over every LLM turn.

    Subclasses implement the four phases; `run` wires them together and
    returns the vote plus the full internal trace for the audit ledger.
    """

    def __init__(self, agent_id: str, config: dict | None = None):
        self.agent_id = agent_id
        self.config = config or {}

    @abstractmethod
    def monitor(self, input_text: str, output_text: str, context: dict) -> dict:
        """Observe the turn through this agent's dimensional lens."""

    @abstractmethod
    def analyse(self, observations: dict) -> dict:
        """Compute deviation from the expected acceptable envelope."""

    @abstractmethod
    def plan(self, analysis: dict) -> dict:
        """Decide a vote type and deviation score from the analysis."""

    @abstractmethod
    def execute(self, plan: dict, analysis: dict) -> AgentVote:
        """Construct the structured AgentVote for this turn."""

    def run(
        self, input_text: str, output_text: str, context: dict | None = None
    ) -> tuple[AgentVote, MAPEKState]:
        context = context or {}
        observations = self.monitor(input_text, output_text, context)
        analysis = self.analyse(observations)
        decision = self.plan(analysis)
        vote = self.execute(decision, analysis)
        mapek_state = MAPEKState(
            monitor=observations,
            analyse=analysis,
            plan=decision,
            execute=vote.model_dump(mode="json"),
            knowledge_refs=[self.agent_id],
        )
        return vote, mapek_state
