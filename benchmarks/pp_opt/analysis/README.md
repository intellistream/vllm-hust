# PP optimization evidence analysis

These scripts perform offline checks over captured pipeline-parallel state-flow
artifacts. They provide derived correctness and layout evidence; they do not
constitute hardware performance results.

Run the unit tests with:

```bash
python -m pytest \
  benchmarks/pp_opt/analysis/test_verify_state_invariants.py \
  benchmarks/pp_opt/analysis/test_verify_paired_layout.py -q
```

Verify scheduler/worker KV ownership invariants for one captured run:

```bash
python benchmarks/pp_opt/analysis/verify_state_invariants.py \
  RUN_DIRECTORY OUTPUT_REPORT.json
```

Build manifests for a baseline and candidate run, then compare them:

```bash
python benchmarks/pp_opt/analysis/verify_paired_layout.py \
  manifest BASELINE_RUN baseline-manifest.json
python benchmarks/pp_opt/analysis/verify_paired_layout.py \
  manifest CANDIDATE_RUN candidate-manifest.json
python benchmarks/pp_opt/analysis/verify_paired_layout.py \
  compare baseline-manifest.json candidate-manifest.json comparison.json
```

Raw run artifacts are intentionally kept outside Git. Issue #147 records the
fixed revisions, report hashes, evidence limitations, and the required next
hardware run.
