"""Conservative helpers for proposing, but never auto-activating, procedures."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SUCCESS_CUE = re.compile(
    r"\b(?:funcionou|deu certo|deu tudo certo|resolveu|resolvido|"
    r"funciona|funcionando|agora funciona|era isso|perfeito|works|working|"
    r"worked|that worked|it worked|"
    r"fixed it|that fixed it|solved|problem solved|success)\b",
    re.IGNORECASE,
)
_FAILURE_CUE = re.compile(
    r"\b(?:"
    r"(?:nao|não) funcionou|(?:nao|não) deu certo|(?:nao|não) resolveu|"
    r"(?:nao|não) funciona|(?:nao|não) está funcionando|"
    r"continua falhando|segue falhando|ainda falha|falhou|"
    r"didn['’]?t work|did not work|doesn['’]?t work|does not work|"
    r"not working|still fails?|still failing|still broken|not fixed|"
    r"failed|failure|problem persists"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_FAILURE = re.compile(
    r"\b(?:nao|não|not|never|didn['’]?t|did not|doesn['’]?t|does not)\b"
    r"(?:[\W_]+\w+){0,2}[\W_]+(?:fail(?:ed|ing)?|failure|broken)\b",
    re.IGNORECASE,
)
_NEGATED_SUCCESS = re.compile(
    r"\b(?:nao|não|not|never|didn['’]?t|did not|ainda nao|ainda não|"
    r"nao sei se|não sei se|not sure if|unsure whether)\b"
    r"(?:[\W_]+\w+){0,3}[\W_]+"
    r"(?:funcionou|funciona|funcionando|deu certo|resolveu|resolvido|"
    r"works|working|worked|fixed|solved|success)\b",
    re.IGNORECASE,
)
_UNCERTAIN_SUCCESS = re.compile(
    r"\b(?:talvez|maybe|acho que|parece que|nao tenho certeza|"
    r"não tenho certeza|not sure|unsure|whether|será que|sera que)\b",
    re.IGNORECASE,
)
_STEP_MARKER = re.compile(
    r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+|(?:step|etapa)\s+\d{1,2}[:.)]\s+)",
    re.IGNORECASE,
)
_MAX_DRAFT_CHARS = 8_000
_PROCEDURE_STOPWORDS = frozenset(
    "a an and as at com da das de do dos e em for from how i in is it na nas no "
    "nos not o of on os para por the to um uma uns umas use usar voce você eu "
    "meu minha meus minhas como qual quais what when where why with please "
    "can could should would me my our your this that these those repair problema "
    "problem ajuda help fix corrigir resolver solve consertar arrumar"
    .split()
)


@dataclass(frozen=True, slots=True)
class ProcedureCandidate:
    """A review-only procedure proposal sourced from one successful episode."""

    skill_name: str
    title: str
    description: str
    draft_markdown: str
    success_signal: str


@dataclass(frozen=True, slots=True)
class ProcedureProposal:
    """Persisted review state for one procedure candidate."""

    proposal_id: str
    evidence_episode_id: str
    session_id: str
    skill_name: str
    title: str
    description: str
    draft_markdown: str
    success_signal: str
    status: str
    skill_path: str
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class RelatedProcedureProposal:
    """A possible cross-session match shown for manual comparison only."""

    proposal: ProcedureProposal
    similarity: float


@dataclass(frozen=True, slots=True)
class ProcedureOutcome:
    """An explicit user-reported result linked to a preceding procedure episode."""

    outcome_id: str
    evidence_episode_id: str
    session_id: str
    task_title: str
    outcome: str
    signal: str
    created_at: float


@dataclass(frozen=True, slots=True)
class RelatedProcedureOutcome:
    """A possible outcome match for a similar task, not proof of correctness."""

    outcome: ProcedureOutcome
    similarity: float


def procedure_title_similarity(left: str, right: str) -> float:
    """Return lexical Jaccard similarity for two short procedure request titles."""
    def tokens(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", (value or "").casefold())
        ascii_text = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return {
            token
            for token in re.findall(r"[a-z0-9]+", ascii_text)
            if token not in _PROCEDURE_STOPWORDS and len(token) > 1
        }

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return 0.0
    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return 0.0
    union = left_tokens | right_tokens
    return len(shared) / len(union) if union else 0.0


def is_explicit_success(text: str) -> bool:
    """Require a direct, non-negated user success cue before proposing anything."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if (
        not normalized
        or "?" in normalized
        or not _SUCCESS_CUE.search(normalized)
        or _UNCERTAIN_SUCCESS.search(normalized)
    ):
        return False
    return _NEGATED_SUCCESS.search(normalized) is None


def is_explicit_failure(text: str) -> bool:
    """Require a direct failure report, rejecting uncertainty and mixed outcomes."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if (
        not normalized
        or "?" in normalized
        or _UNCERTAIN_SUCCESS.search(normalized)
        or not _FAILURE_CUE.search(normalized)
        or _NEGATED_FAILURE.search(normalized)
    ):
        return False
    success_matches = list(_SUCCESS_CUE.finditer(normalized))
    negated_matches = list(_NEGATED_SUCCESS.finditer(normalized))
    if any(
        not any(
            negative.start() <= success.start()
            and negative.end() >= success.end()
            for negative in negated_matches
        )
        for success in success_matches
    ):
        return False
    return True


def has_structured_procedure(assistant_text: str) -> bool:
    """Check that an exchange contains at least two marked procedure steps."""
    return bool(_procedure_lines(assistant_text))


def _procedure_lines(assistant_text: str) -> str:
    """Keep only a clearly structured list, excluding surrounding dialogue."""
    text = (assistant_text or "").strip()
    lines = text.splitlines()
    markers = [index for index, line in enumerate(lines) if _STEP_MARKER.match(line)]
    if len(markers) < 2:
        return ""
    selected: list[str] = []
    in_list = False
    for line in lines:
        if _STEP_MARKER.match(line):
            in_list = True
            selected.append(line)
        elif in_list and (not line.strip() or line[:1].isspace()):
            selected.append(line)
        elif in_list and line.strip():
            in_list = False
    return "\n".join(selected)[:_MAX_DRAFT_CHARS].strip()


def _title_from_request(user_text: str) -> str:
    title = re.sub(r"\s+", " ", (user_text or "").strip())
    title = re.sub(r"^[#>*\-\s]+", "", title)
    if len(title) > 160:
        title = title[:157].rstrip() + "..."
    return title or "Successful procedure"


def build_procedure_candidate(
    *,
    episode_id: str,
    user_text: str,
    assistant_text: str,
    success_signal: str,
) -> ProcedureCandidate | None:
    """Build a draft only from explicit success plus a multi-step assistant reply.

    Only the assistant's marked list items are copied into review material; this
    function does not reinterpret them into executable tool calls or authorize
    a skill.
    """
    if not is_explicit_success(success_signal):
        return None
    draft = _procedure_lines(assistant_text)
    if not draft:
        return None

    safe_id = re.sub(r"[^a-f0-9]", "", (episode_id or "").lower())[:16]
    if len(safe_id) < 8:
        return None
    title = _title_from_request(user_text)
    description = (
        "User-reviewed routine. Invoke only when the user explicitly asks to use "
        "this skill by name; review every instruction before acting."
    )
    return ProcedureCandidate(
        skill_name=f"ace-procedure-{safe_id}",
        title=title,
        description=description,
        draft_markdown=draft,
        success_signal=(success_signal or "").strip()[:1_000],
    )


def skill_markdown(
    candidate: ProcedureCandidate | ProcedureProposal,
    evidence_episode_id: str,
    *,
    version: str = "0.1.0",
    skill_name: str | None = None,
) -> str:
    """Render an explicitly approved draft as a user-invoked skill."""
    import json

    name = skill_name or candidate.skill_name
    if not re.fullmatch(r"ace-procedure-[a-f0-9]{8,16}", name):
        raise ValueError("Skill name must identify an ACE procedure.")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Skill version must use MAJOR.MINOR.PATCH.")
    description = json.dumps(candidate.description, ensure_ascii=False)
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"version: {version}\n"
        "author: A.C.E.\n"
        "tags: [ace, reviewed-proposal]\n"
        "user_invocable: true\n"
        "disable_model_invocation: false\n"
        "---\n\n"
        f"# {candidate.title}\n\n"
        "This procedure was proposed from a conversation and explicitly approved "
        "by the user. Its source is historical text. Apply it only when the user "
        "explicitly asks to use this skill by name and the steps still fit the "
        "request. Keep normal tool permissions and confirmations in force.\n\n"
        "## Draft steps to review\n\n"
        f"{candidate.draft_markdown}\n\n"
        "## Provenance\n\n"
        f"Source episode: `{evidence_episode_id}`\n"
    )


__all__ = [
    "ProcedureCandidate",
    "ProcedureOutcome",
    "ProcedureProposal",
    "RelatedProcedureOutcome",
    "RelatedProcedureProposal",
    "build_procedure_candidate",
    "has_structured_procedure",
    "is_explicit_failure",
    "is_explicit_success",
    "procedure_title_similarity",
    "skill_markdown",
]
