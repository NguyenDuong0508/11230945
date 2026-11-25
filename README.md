# A Big-Data Pipeline for Bank Failure Early Warning

**BFT65B — Big Data Analytics in Finance | National Economics University | Student ID: 11230945**

## Part 1 — Acquisition & Storage

This repository acquires the FDIC BankFind source tables, verifies their completeness, and evaluates storage formats and layouts using controlled measurements. It establishes the immutable Bronze data layer for the subsequent data-quality audit, bank–quarter panel and early-warning study.

### Scope and design

- Financials: all insured institutions for every calendar quarter from 1984Q1 through 2026Q2 (170 quarters). No active-bank or outcome-based filter.
- Failures and Institutions: complete endpoint populations with selected source fields. Failure records include assistance transactions; `RESTYPE`, `RESTYPE1`, `FIN`, `FAILDATE` and cost provenance are retained, without outcome recoding. No records are dropped for missing failure CERTs.
- Financials are written to typed Snappy Parquet, partitioned by calendar year and quarter. Reference tables are stored separately. Each artifact has a checksum, row count, schema contract and API pagination evidence. The layout is provisional until the storage experiment has been evaluated.
- Acquisition verifies completeness against each request's reported API total, all expected quarters, physical files, required keys and read-back checks. A live API is not a versioned historical snapshot; unchanged totals alone cannot establish transactional consistency across an extended download. Retrieval timestamps and file hashes identify the acquired data vintage.
- Selected source values are preserved except explicit type conversion: `REPDTE` becomes a calendar date, numeric fields are converted to typed numbers (blank numeric values become null), reference dates remain text for later date-quality adjudication. Zeros and negative amounts are not corrected at this stage. An invalid key or unparsable numeric field stops acquisition rather than silently dropping a row.

See [field selection and semantics](docs/field_contract.md) and [downstream interface and research constraints](docs/part1_protocol.md).

## Run in Google Colab (CPU)

Use a fresh Python runtime and run the notebooks in order. Each notebook obtains the current repository code, installs the pinned dependencies in subprocesses, runs the mandatory offline test suite, and displays full failure diagnostics. No API key is required.

1. [01 — Acquisition and Verification](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/01_acquisition_colab.ipynb). Mount Google Drive; use `MyDrive/11230945/` as the workspace. Retrieve one initial quarter as an acquisition check and then resume until all 170 quarters and both reference tables are verified. A partial download is not an accepted Bronze run.
2. [02 — Storage Experiments](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/02_storage_experiments_colab.ipynb). Use the same workspace. The notebook resolves the verified source manifest; no upstream run identifier is entered manually. All storage variants are staged to `/content` local storage. Inspect the measured results, equivalence checks, size and candidate-fragment counts.
3. Save the executed notebooks, including the verification summaries, measurements and figures, to GitHub at their original paths. Do not commit source data, local scratch files, or large raw outputs. The committed templates initially contain no empirical results; execute them to produce the required evidence.

## Run from a Linux checkout

Python 3.11–3.13, with a working internet connection for dependencies and FDIC. Use a dedicated virtual environment. The benchmark uses Linux process RSS units and requires Snappy and ZSTD codec support. `--workspace` and `--local` must be separate directories; the benchmark uses local disk only for timed reads.

```bash
git clone https://github.com/NguyenDuong0508/11230945.git
cd 11230945
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --junitxml=/tmp/part1-tests.xml
python -c "import xml.etree.ElementTree as E; s=E.parse('/tmp/part1-tests.xml').findall('.//testsuite'); assert s and sum(int(t.get('skipped',0)) for t in s)==0"
python run_part1.py acquire --workspace "$HOME/fdic_workspace" --max-new-quarters 1
python run_part1.py acquire --workspace "$HOME/fdic_workspace"
python run_part1.py verify --workspace "$HOME/fdic_workspace"
python run_part1.py benchmark --workspace "$HOME/fdic_workspace" --local /tmp/fdic_benchmark --warmups 1 --repetitions 5 --threads 2
```

All tests in `tests/` must pass, without skips. Tests use synthetic FDIC-shaped data; they do not download the live source. Use `python -m pytest tests/unit -q` only for dependency-light debugging, not as a substitute for the full integration test gate. GitHub Actions runs the complete suite on Python 3.11 and 3.12. A test failure is a stop condition; inspect the traceback and resolve the failure before acquisition.

The acquisition command resumes a pending run, or verifies an already sealed run. To collect a new data vintage, use `acquire --fresh` exactly once, then continue with the ordinary command. Do not run simultaneous writers against the same workspace. A corrupt unsealed checkpoint is downloaded again; a modified sealed run fails verification and is never silently rewritten. The active source pointer is updated only after verification. The storage benchmark refuses to proceed while a pending acquisition exists. A failed benchmark leaves scratch files for diagnosis; a successful benchmark copies checksummed evidence to the workspace and removes only its own scratch directory.

## Storage experiment

| Representation | Format | Layout | Codec |
|---|---|---|---|
| `csv` | Uncompressed CSV | Flat | None |
| `flat_snappy` | Parquet | Flat | Snappy |
| `flat_zstd` | Parquet | Flat | ZSTD level 3 |
| `partitioned_snappy` | Parquet | Year/quarter | Snappy |
| `partitioned_zstd` | Parquet | Year/quarter | ZSTD level 3 |

Workloads: full scan (all fields); projection (six fields); the same projection for 2020; the same projection for 2020Q1. For partitioned Parquet, the date filter is combined with matching Hive partition keys. CSV uses the same typed projection but must scan/parse the file. All five representations derive from the same verified data; full round-trip values and all workload outputs are compared by key before benchmarking. CSV nullability is restored only after required-field and type validation. Float comparisons allow small numerical representation differences; null masks and keys must match exactly.

The default experiment contains 20 format/workload pairs and a separate pandas–Polars CSV projection experiment (two pairs). One warm-up and five measurements per pair give 22 warm-ups and 110 measured trials (132 worker processes). Warm-ups precede measurements; order is randomized within each block with seed 11230945. Each measurement uses a fresh worker, consistent thread controls where supported, and the same local filesystem. Library import and process startup are outside timing; timed materialization includes scan metadata and schema restoration. Peak RSS is whole-worker memory including imports. The OS page cache is not flushed, so this is not a guaranteed cold-cache experiment. The CSV engine experiment uses identical four selected columns and string dates, and is reported separately from the Arrow format comparison.

Use medians and IQRs, not a single fastest trial. File candidate counts demonstrate partition pruning, not actual bytes transferred; flat Parquet can still skip row groups. The production recommendation should consider measured size, latency and peak memory. Do not claim a codec or partitioning is faster without corresponding results. Colab local CPU measurements do not establish multi-node Spark break-even.

## Workspace and handoff

```
workspace/
├── active_source.json             # relative path and verified manifest SHA-256
├── pending_source.json            # present only while an acquisition is pending
├── bronze/<run-id>/
│   ├── manifest.json
│   ├── verification.json
│   ├── financials/year=YYYY/quarter=Q/part.parquet
│   ├── failures.parquet
│   └── institutions.parquet
├── active_storage.json            # evidence pointer and source/evidence hashes
└── storage_evidence/<run-id>/
    ├── preparation.json
    ├── trials.csv
    ├── summary.csv
    ├── sizes.csv
    ├── checks.json
    ├── storage_size.png
    ├── latency.png
    ├── memory.png
    └── evidence.json
```

Part 2 resolves `active_source.json`, verifies the manifest SHA-256 and revalidates the actual Parquet files. Part 3 can use the storage evidence together with the clean panel from Part 2. Part 4 derives features and horizon-specific labels from Part 2, not from benchmark replicas. No target, test selection, imputation, outlier correction or terminal-event filtering occurs in Part 1. The test split and modelling policy must be specified independently in later research stages.

## Sources and disclosure

- FDIC, [BankFind Suite API documentation](https://api.fdic.gov/banks/docs/), [financial field definitions](https://api.fdic.gov/banks/docs/risview_properties.yaml), [failure field definitions](https://api.fdic.gov/banks/docs/failure_properties.yaml), [institution field definitions](https://api.fdic.gov/banks/docs/institution_properties.yaml), and [bulk data](https://banks.data.fdic.gov/bankfind-suite/bulkData).
- National Economics University, BFT65B final examination instructions and `fdic_download_starter.py` (course materials): dataset scope, field selection starting point, quarter-by-quarter API retrieval and completeness requirements.
- Apache Arrow, [CSV documentation](https://arrow.apache.org/docs/python/csv.html), [Parquet documentation](https://arrow.apache.org/docs/python/parquet.html), [Dataset documentation](https://arrow.apache.org/docs/python/dataset.html); [pandas](https://pandas.pydata.org/docs/), [Polars](https://docs.pola.rs/), [Requests](https://requests.readthedocs.io/), [pytest](https://docs.pytest.org/), [Matplotlib](https://matplotlib.org/stable/).
- OpenAI ChatGPT assisted in designing, writing and reviewing acquisition, storage experiments, financial data quality, event-policy tests, scalable processing experiments and documentation. Empirical results must be obtained by executing the pipeline, and the submitting researcher must independently verify the code, results and interpretation. Cite this assistance in the final report as required by the examination.


## Part 2 — Data Quality and Panel

[Notebook 03: Data Quality and Panel](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/03_quality_panel_colab.ipynb) consumes the sealed Bronze source from Notebook 01. It does not download data or modify the source. Run all tests (including the new Part 2 integration test) before processing. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_part2.py build --workspace /path/to/workspace
python run_part2.py verify --workspace /path/to/workspace
```

Use the same workspace as Part 1; on Colab this is `/content/drive/MyDrive/11230945`. `active_panel.json` points to a verified `silver/<run-id>/` containing `panel/` (quarter-partitioned Parquet), `event_ledger.parquet`, `terminal_audit.parquet`, quality/evidence CSVs, figures and a manifest. Financial records are retained, reporting gaps are not filled, and the event and retrospective audit tables are kept separate. See [Part 2 protocol](docs/part2_protocol.md) for treatment rules, lag assumptions, capital and income sensitivity diagnostics, FDIC event definitions, limitations and the downstream interface. Part 3 and Part 4 resolve and verify the active Silver pointer; never use full-history terminal audit status as a blanket filter or historical predictor. No model split or final labels are created here.

Additional sources: [FDIC Failures and Assistance definitions](https://banks.data.fdic.gov/bankfind-suite/failures/help); [FDIC Disclaimers and Methodologies](https://banks.data.fdic.gov/bankfind-suite/help?helpTopic=disclaimers-and-methodologies); [FDIC Community Banking Research Program](https://www.fdic.gov/community-banking-research-program) (CBLR reporting). These references motivate the event and reporting-regime policies documented in Part 2.


## Part 3 — Scalable Processing

[Notebook 04: Scalable Processing](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/04_scalable_processing_colab.ipynb) reads the verified Silver pointer. Install Java 17 or later (set `JAVA_HOME` if required), then run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_part3.py run --workspace /path/to/workspace --scratch /path/to/local/scratch
python run_part3.py verify --workspace /path/to/workspace
```

The controlled eight-column projection is measured at 1× (all observed rows), 5× and 10× (synthetic replicated histories with disjoint CERTs). Three analytical workloads compare pandas and Polars at all scales and PySpark on 1×. One additional Polars single-thread scenario per workload measures thread-count sensitivity. Full parity, timings, sampled process-tree RSS, optimized and physical plans, environment and staging evidence are saved under `processing_evidence/<run-id>/`. `active_processing.json` points to the verified evidence. Synthetic data remain in local scratch and are not model inputs. See [Part 3 protocol](docs/part3_protocol.md) for experiment design, memory and timing boundaries, Amdahl interpretation and cluster-decision limitations. Part 4 continues from the verified original Silver data. Sources: [Polars lazy optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/), [Spark SQL performance tuning](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html), [PySpark installation](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html), BFT65B course materials.
