---
target: docs/index.html
total_score: 22
p0_count: 0
p1_count: 2
timestamp: 2026-06-14T07-57-19Z
slug: docs-index-html
---
# Impeccable Critique: docs/index.html

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|---:|---:|---|
| 1 | Visibility of System Status | 2 | Model load is visible, but the post-load state says "新对局" while the board has no actual game state until the user clicks it. |
| 2 | Match System / Real World | 2 | "深算", "策略", "后续路线", and "MCTS" are partially explained, but the user still has to translate them into a move decision. |
| 3 | User Control and Freedom | 2 | New game, undo, hint, and overlays exist; side choice currently appears constrained to "我执白（后手）", which reads like a disabled missing option. |
| 4 | Consistency and Standards | 3 | Controls and panels are cohesive; the status pill mixes app state and action result language. |
| 5 | Error Prevention | 2 | The blank-but-ready initial state makes misclicks likely; recovery depends on the user understanding "新对局" as the next step. |
| 6 | Recognition Rather Than Recall | 2 | Labels are visible, but interpreting table metrics, board badges, tree branches, and win chart requires recall. |
| 7 | Flexibility and Efficiency | 2 | Slider and overlay modes help; no keyboard shortcuts or fast analysis presets are exposed. |
| 8 | Aesthetic and Minimalist Design | 3 | Calm, board-first product UI; analysis surfaces become dense after hint/search. |
| 9 | Error Recovery | 2 | Toast/status feedback exists, but failure states are transient and not always actionable. |
| 10 | Help and Documentation | 2 | Inline definitions exist; there is no compact "how to read this recommendation" help at the moment of need. |
| **Total** |  | **22/40** | **Acceptable: solid foundation, but the analysis experience needs sharper hierarchy.** |

## Anti-Patterns Verdict

**LLM assessment:** This does not scream "AI made this." The board-first layout, restrained palette, and compact product controls are credible. The remaining tell is not decorative slop; it is product indecision. The interface exposes multiple internal model artifacts at equal weight: search heat, policy probability, principal variation numbers, candidate rows, win chart, MCTS tree, and move history. That makes it feel like a debug dashboard instead of a thinking partner.

**Deterministic scan:** CLI detector found 3 warnings, all in `docs/styles.css`: layout property animation at line 285 (`transition: width`), line 321 (`transition: height`), and line 752 (`transition: width`). Browser overlay injection succeeded in the main thread and reported 2 anti-pattern categories: layout property animation and low contrast text. The low-contrast finding did not appear in the CLI output, so treat muted helper labels as a visual QA target.

**Visual overlays:** Main-thread overlay injection succeeded on the local page. Console reported "[impeccable] 2 anti-patterns found", and DOM evidence showed overlay labels for "layout property animation" and "low contrast text". Sub-agent overlay failed because IAB visibility is not supported inside a subagent thread, but the main-thread retry produced usable browser evidence.

## Overall Impression

The product is close to a good AlphaZero teaching/play surface, but the hint mode currently answers "here is everything the model knows" instead of "here is the move worth considering, and why." The largest opportunity is to make one recommendation object primary and let tree/table/detail unfold around it.

## What's Working

1. The board has the right priority. It is large, quiet, readable, and avoids toy-game styling.
2. The product register is right: calm, analytical, and not overly decorative.
3. The split between plain board, search, and policy overlays is conceptually strong.

## Priority Issues

### [P1] The initial state looks ready but is not a game yet

**Why it matters:** The page says the model is ready and status is "新对局", but the turn, evaluation, candidates, and tree are blank. A first-time user can click the board, get no meaningful progress, and assume the app is broken.

**Fix:** After load, either automatically start the default game or make "新对局" unmistakably the primary next action with a short empty-state prompt near the board. Status should read like a state ("等待开始") and the button should read like an action ("开始对局").

**Suggested command:** `$impeccable harden docs`

### [P1] Hint mode overwhelms the actual move decision

**Why it matters:** After hint/search, the user must parse heat labels, PV badges, win rate, chart, table, tree, and history. The product promise is understanding the model, but the UI makes users assemble the answer themselves.

**Fix:** Add a compact recommendation strip below or beside the board: "建议 3,5: 搜索访问最高；风险：局面胜率低于另一候选。" Then make candidate table and MCTS tree secondary detail.

**Suggested command:** `$impeccable distill docs`

### [P2] Search/policy/PV copy is technically correct but not decision-oriented

**Why it matters:** "策略=模型自学出的落点概率" is accurate, but users still ask what it means for their next move. The terms need to explain trust and usage, not only definition.

**Fix:** Rewrite microcopy into operational language: "深算：AI 想过之后重点看的点", "策略：搜索前模型最自然的选择", "后续路线：如果双方都按搜索主线走，接下来可能发生什么."

**Suggested command:** `$impeccable clarify docs`

### [P2] MCTS tree is large but low-yield

**Why it matters:** The tree now has more room below the board, but it still behaves like a static graph. It does not clearly explain depth, branch pruning, or why a branch is emphasized. On desktop it starts below the fold; on mobile it appears before core controls.

**Fix:** Make the default tree a principal-line ladder plus depth bands. Add "显示完整树" as an advanced view. Use branch thickness for visit share, depth lanes for search depth, and a clear current-node-to-recommendation path.

**Suggested command:** `$impeccable layout docs`

### [P3] Mobile order puts deep analysis before controls

**Why it matters:** At a 390px-wide viewport, the MCTS tree begins before "对局设置"; the user has to scroll past analysis before primary controls and stats.

**Fix:** Mobile order should be board, actions, recommendation summary, evaluation, candidates, tree, history.

**Suggested command:** `$impeccable adapt docs`

### [P3] Motion uses layout property transitions

**Why it matters:** Width/height transitions can cause jank during frequent model updates and are specifically flagged by the detector.

**Fix:** Replace width/height meter transitions with transform-scale based fills or CSS custom properties that animate transforms.

**Suggested command:** `$impeccable optimize docs`

## Persona Red Flags

**Alex, power user:** No visible shortcuts for new game, undo, hint, or overlay switching. The simulation slider is useful, but there is no quick low/medium/high preset for repeat play.

**Jordan, first-timer:** The first screen does not say "start here." AI search terms are visible but still feel like labels from an engine UI, not advice.

**Sam, accessibility-dependent user:** Focus styles exist and board cells are buttons, which is good. But heat intensity, line thickness, chart motion, and color-coded row quality need stronger text equivalents.

**Casey, mobile user:** The board fits on phone, but the page becomes a long vertical stack. Primary actions should come before the MCTS tree.

## Minor Observations

- The candidate table can show an apparent conflict: the selected row may have the highest visit share while another candidate has higher displayed win rate. That needs a plain-language explanation.
- The evaluation bar beside the board is useful after analysis, but cryptic in the empty state.
- The browser rendered Chinese correctly; source preview mojibake seen in one shell read was an encoding display issue, not a page rendering defect.
- The MCTS tree is currently capped by implementation: exported visualization has `maxDepth = 5`, with side branches aggressively pruned. The UI should explain that it is a summarized search tree, not the complete tree.

## Questions to Consider

1. What if the first post-hint surface always answered: "where should I play next, and why?"
2. Should the tree be a teaching object by default, or an advanced inspection mode?
3. Should "策略" stay as a technical term, or become a more human-facing "模型倾向" label?
