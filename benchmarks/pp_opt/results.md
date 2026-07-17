# PP Optimization Results

## Validated Qwen3-32B comparison

The validated experiment used Qwen3-32B, PP4 + TP2, eight Ascend 910C NPUs,
the 1,000-request conversation trace, four static microbatches, calibrated
cost placement, full transformer execution, and no PP send overlap. Both modes
completed all requests and generated the same 349,357 output tokens.

| Mode | Duration | Request throughput | Output throughput |
| --- | ---: | ---: | ---: |
| Baseline | 1,446.91 s | 0.6911 req/s | 241.45 tok/s |
| PP-opt | 1,187.73 s | 0.8419 req/s | 294.14 tok/s |
| Speedup | **1.218x** | **+21.82%** | **+21.82%** |

This is the only full end-to-end pair currently treated as a validated result.
The remaining Qwen3-32B BurstGPT and Qwen3-235B pairs must be rerun with the
same recovered static configuration before publication.

## Pipeline compactness

A matched five-second CANN profile used the same static configuration under
approximately 90 active requests and 90-95% KV-cache occupancy. Metrics use
TP rank zero for each PP stage.

| Metric | Baseline | PP-opt | Change |
| --- | ---: | ---: | ---: |
| Average PP stages computing | 0.887 | 1.710 | **+92.81%** |
| Mean per-stage compute duty | 22.18% | 42.76% | **+92.81%** |
| Time with at least 2 stages computing | 0.00% | 54.70% | **+54.70 pp** |
| Time with all 4 stages computing | 0.00% | 6.85% | **+6.85 pp** |
| Median wave start skew | 433.5 ms | 179.5 ms | **-58.60%** |
| P95 wave span | 764.1 ms | 292.4 ms | **-61.73%** |
| Active-kernel Cube utilization | 76.67% | 55.12% | -28.11% |
| Wall-clock Cube utilization proxy | 13.00% | 15.65% | **+20.40%** |
| Time with no stage computing | 11.29% | 14.85% | +3.57 pp |

The baseline produced 19 large forward waves in the capture window, with a
median 90 requests and 1.339 million context tokens per wave. PP-opt produced
76 waves, exactly four times as many, with medians of 21.5 requests and 341
thousand context tokens. Multiple stages computed concurrently for 54.7% of
the optimized window versus 0% of the baseline window.

The concurrency gain is larger than the end-to-end speedup because smaller
microbatches reduced active-kernel Cube utilization by 28.1%. Wall-clock Cube
utilization still rose by 20.4%, closely matching the measured 21.8% output
throughput improvement.

The machine-readable profile summary and figure are under
`results/pipeline_profile/analysis/`. The Cube metric is an MFU proxy derived
from CANN `PipeUtilization`; it is not a formal FLOP-based MFU value.

## Calibration quality

Each checked-in deployment was profiled over 48 workload points. The installed
models are hash-bound to their model and benchmark configurations.

| Model | Samples per rank/cost | R2 range | Configuration |
| --- | ---: | ---: | --- |
| Qwen3-32B | 3,852-3,877 | 0.925-0.934 | `configs/qwen3_32b_pp4tp2_8x910c/` |
| Qwen3-235B-A22B | 3,879-3,956 | 0.626-0.660 | `configs/qwen3_235b_pp4tp2_8x910c/` |

The 235B fit has materially lower R2 and must pass the short end-to-end gate
before any full result is accepted.
