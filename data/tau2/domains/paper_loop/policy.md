# Paper-loop task policy

## Scope

Each task is an independent, paper-backed software reconstruction case. Use
the case's `paper.pdf` as primary evidence, follow `TASK.md`, and begin with a
fresh copy of `starter/`. The starter is editable input, not a reference
answer.

## Required behavior

- Distinguish statements supported by the paper from engineering inferences
  and substitutes introduced only to make the implementation runnable.
- Keep the implementation, execution graph, explanation, and actual output
  mutually consistent.
- Use Python's standard library and NumPy, run on CPU without network access,
  and make fixed-seed runs deterministic.
- Preserve enough raw output for an external evaluator to recompute objectives
  and constraints. A candidate's self-reported pass flag is not evidence.
- Explain unavailable data, maps, weights, training artifacts, or proprietary
  tool outputs, including how each substitute affects the result.

## Evaluation boundary

The case bundle contains only agent-visible inputs. A compatible evaluator is
expected to create an isolated writable workspace, rerun public deterministic
checks, verify that the submitted snapshot is unchanged during evaluation,
and apply an independently maintained semantic oracle. Evaluator and oracle
material must not be added to the agent-visible case directory.

Historical development records are provenance, not task inputs, and must not
be injected into a fresh trial.

## Lifecycle and splits

`ready` cases completed the earlier loop and may appear in `base`. `incomplete`
cases remain train-only. `candidate` cases are reserved for `test` until an
external evaluator freezes and validates their independent oracles. Split
membership alone does not change lifecycle status.
