"""Pydantic v2 schemas shared by every 5P agent and the consensus layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class VoteType(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    VETO = "veto"


class AgentVote(BaseModel):
    agent_id: str
    vote: VoteType
    deviation_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MAPEKState(BaseModel):
    monitor: dict = Field(default_factory=dict)
    analyse: dict = Field(default_factory=dict)
    plan: dict = Field(default_factory=dict)
    execute: dict = Field(default_factory=dict)
    knowledge_refs: list[str] = Field(default_factory=list)


class ConflictType(str, Enum):
    TYPE_A = "performance_vs_parity"
    TYPE_B = "proxies_vs_participation"
    TYPE_C = "process_vs_parity"


class Conflict(BaseModel):
    type: ConflictType
    agents: tuple[str, str]
    description: str


class TurnState(BaseModel):
    turn_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_text: str
    output_text: str
    agent_votes: list[AgentVote] = Field(default_factory=list)
    mapek_states: dict[str, MAPEKState] = Field(default_factory=dict)
    consensus: str
    conflicts: list[Conflict] = Field(default_factory=list)
    metacognitive_summary: str
    known_unknowns: list[str] = Field(default_factory=list)
