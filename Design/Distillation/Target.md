# Distillation Targets

## Phase 1 - Raw KataGo teacher

Distillation will be done on the KataGo teacher net without meaningful search. Use
KataGo's analysis engine with `maxVisits: 1` as the export surface, but prefer the
raw neural fields over search-shaped move statistics wherever the response exposes
them.

Recommended query options:

- `includePolicy: true` so the response includes the full raw policy.
- `includeOwnership: true` so the response includes the root ownership map.
- `includeNoResultValue: true` if no-result value is trained.
- `analysisPVLen: 0` for throughput when principal variations are not needed.
- `maxVisits: 1` for Phase 1.

Response fields to treat as the Phase 1 teacher contract:

- `policy`: length `boardYSize * boardXSize + 1`; row-major from top-left to
  bottom-right, with pass as the final entry. Legal probabilities are positive
  and sum to 1; illegal board moves are `-1`.
- `ownership`: length `boardYSize * boardXSize`; row-major from top-left to
  bottom-right; values are in `[-1, 1]`.
- `rootInfo.rawWinrate`, `rootInfo.rawLead`, and `rootInfo.rawNoResultProb`:
  preferred value targets when present. Fall back to `rootInfo.winrate` and
  `rootInfo.scoreLead` only when raw fields are absent.
- `moveInfos[*].prior`: useful for audit, but not the primary policy target when
  the full `policy` array is available.

KataGo analysis values are reported from the perspective selected by
`reportAnalysisWinratesAs` in the analysis config. The local bundled config uses
`BLACK`, so targets must be converted to SAKIGo's current-player perspective when
the side to move is white.

The public analysis protocol does not expose every raw tensor head in the KataGo
model. In particular, do not assume access to the opponent-next-policy head or the
full score-distribution head unless we add a deeper KataGo reader.

## Phase 2 - High-visit teacher

The net will be fine tuned using high-visit data. In this phase, `moveInfos`,
visit counts, high-visit `rootInfo`, and optional per-move ownership become useful
training/audit signals rather than merely a wrapper around the raw teacher net.
