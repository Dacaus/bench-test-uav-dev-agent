# UAV paper-loop case bundle

This repository is a case-only dataset containing 22 paper-backed UAV and
optimization tasks. It follows the Tau task-data convention of keeping a
domain's task definitions, split definitions, policy, and task assets under
`data/tau2/domains/<domain>/`.

The repository intentionally contains no benchmark runner, environment
adapter, evaluator, independent oracle, historical trial output, or unrelated
project directory.

## Layout

```text
data/tau2/domains/paper_loop/
├── tasks.json
├── split_tasks.json
├── case_catalog.jsonl
├── policy.md
└── assets/
    ├── 01-ifds-ann/
    │   ├── TASK.md
    │   ├── paper.pdf
    │   ├── manifest.json
    │   └── starter/
    └── ... 22-amiso/
```

Every case is self-contained:

- `TASK.md` describes the requested reconstruction and acceptance criteria.
- `paper.pdf` is the primary evidence for the task.
- `starter/` is the editable starting candidate, not a reference answer.
- `manifest.json` records local paths, lifecycle metadata, and SHA-256
  integrity data.

`tasks.json` is the Tau-style task list. `split_tasks.json` assigns the 22
tasks to `train`, `base`, and `test`. `case_catalog.jsonl` is a public index;
it does not expose evaluator-only oracle data or historical run provenance.

## Status semantics

- `ready`: the earlier development/review loop completed; eligible for `base`.
- `incomplete`: useful development exists, but the loop did not complete.
- `candidate`: reserved as a test case and not yet admitted to `base`.

This repository supplies task inputs only. Scored execution requires a
separate compatible environment and evaluator. The evaluator owns its oracle
and must not treat the starter candidate's `result.json` as ground truth.

To smoke-test a starter locally, install NumPy and run `python3 run.py` from
that case's `starter/` directory.
