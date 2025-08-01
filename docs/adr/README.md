# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Multi-Sensor SNN-Fusion project.

## ADR Template

When creating new ADRs, use the following template:

```markdown
# ADR-XXX: [Short Title]

**Date**: YYYY-MM-DD
**Status**: [Proposed | Accepted | Rejected | Deprecated | Superseded]
**Deciders**: [List of people involved in the decision]

## Context

[Describe the context and problem statement that led to this decision]

## Decision

[Describe the chosen solution]

## Rationale

[Explain why this solution was chosen over alternatives]

## Consequences

[Describe the resulting context after applying the decision, including positive and negative consequences]

## Alternatives Considered

[List and briefly describe other options that were considered]

## References

[Links to supporting documentation, research papers, or related decisions]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-liquid-state-machine-selection.md) | Liquid State Machine Selection | Accepted | 2025-08-01 |
| [002](002-event-driven-processing.md) | Event-Driven Processing Architecture | Accepted | 2025-08-01 |
| [003](003-multi-modal-fusion-strategy.md) | Multi-Modal Fusion Strategy | Accepted | 2025-08-01 |

## ADR Lifecycle

1. **Proposed**: Initial proposal for discussion
2. **Accepted**: Decision has been made and is being implemented
3. **Rejected**: Decision was considered but not adopted
4. **Deprecated**: Decision is no longer recommended but may still be in use
5. **Superseded**: Decision has been replaced by a newer ADR

## Guidelines

- ADRs should be immutable once accepted
- If a decision needs to change, create a new ADR that supersedes the old one
- Include enough context so future team members can understand the decision
- Link to related ADRs when decisions are interconnected
- Use clear, concise language and avoid jargon where possible