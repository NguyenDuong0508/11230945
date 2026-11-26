# A Big-Data Pipeline for Bank Failure Early Warning

**BFT65B — Big Data Analytics in Finance | National Economics University | Student ID: 11230945**

[![Pipeline validation](https://github.com/NguyenDuong0508/11230945/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/NguyenDuong0508/11230945/actions/workflows/tests.yml)

## 1. Research overview

This study develops a reproducible financial data pipeline for bank failure early warning using publicly available FDIC BankFind data. It integrates verified acquisition, efficient storage, financial data-quality assessment, scalable analytical processing, decision-time feature construction and chronological model validation. The primary contribution is the data-engineering and evaluation architecture, rather than model accuracy alone.

The study addresses three research questions:

- **Q1 — Storage and scalability:** How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed?
- **Q2 — Data quality and temporal integrity:** What quality problems occur in the raw data, and how can they be handled without introducing look-ahead bias?
- **Q3 — Early-warning feasibility:** Can the resulting pipeline support an honestly validated early-warning model?

The US dataset provides an empirical setting for studying the methodology. Application to Vietnam requires locally authorized data, institutional definitions, independent validation and supervisory governance; the fitted US model is not directly transferable.

## 2. Principal empirical findings

The following results describe the recorded research snapshot and experiments. Detailed tables, figures, and diagnostics are retained in the executed notebooks.

| Research component | Recorded result |
|---|---|
| Source coverage | 1984Q1–2026Q2; 170 financial report quarters |
| Source population | 1,682,739 Financials rows; 4,117 failure/assistance records; 27,838 Institutions rows |
| Institutional coverage | 24,064 distinct financial CERTs |
| Storage footprint | CSV: 354.45 MiB; flat ZSTD Parquet: 178.60 MiB (49.61% smaller) |
| Filtered storage access | Partitioned Snappy: 87.18× annual-query and 113.87× quarterly-query speedup relative to CSV |
| Original-panel processing | Polars: 1.18–5.26× faster than pandas across the three measured workloads |
| Synthetic scale-up | Up to 16,827,390 logical rows; all measured 10× Polars workloads within 7.94 s; largest sampled peak RSS 3,079 MiB |
| Silver panel | 1,682,739 retained financial observations |
| Gold risk set | 1,640,991 decision-time observations; 1,573,640 labelled; 67,351 censored |
| Model development | Four chronological folds; eight specifications; selected LightGBM with 22 raw predictors; mean fold AP 0.216778 |
| Historical evaluation | 205,672 labelled observations; AP 0.468612; ROC-AUC 0.975284 |
| Capacity-based screening | 87.89% positive-row recall among the highest-scoring 5% of labelled observations per quarter |
| Frozen alarm policy | 4,738 bank-quarter alarms (2.30%); precision 31.70%; positive-row recall 74.84% |
| Event-level detection | 482 of 513 represented events through test horizons (93.96%); 475 of 505 within the test calendar (94.06%) |
| Warning lead time | Median 282.5 days among detected events; 436 of 513 represented events detected at least 90 days ahead (84.99%) |
| Economic scenario | USD 331,937,060 modelled loss; 85.37% lower than the no-alarm policy |
| Automated validation | 142 offline tests passed without skips in the recorded final evidence synthesis |

The computational findings apply to the measured workloads and environment, not arbitrary data sizes or multi-node deployments. Event recall refers to represented failures, not every FDIC source record. Economic loss is a scenario-based decision measure, not realized savings. Historical discrimination does not establish prospective performance or calibration.

## 3. Data sources and study population

Data are obtained from the public [FDIC BankFind API](https://api.fdic.gov/banks/docs/) using a resumable quarter-by-quarter procedure. No API key is required. All insured institutions are requested without an active-bank or outcome-based filter, together with the complete Failures and Institutions endpoint populations (selected fields).

| Data or period | Recorded scope |
|---|---|
| Acquisition started (UTC) | 25 November 2026, 17:36:55 |
| Acquisition sealed (UTC) | 25 November 2026, 17:53:11 |
| Financials | 1984Q1–2026Q2 (170 quarters) |
| Gold decision calendar | 1984Q1–2025Q2 (166 quarters) |
| Training split | 1984Q1–2000Q4; first available labelled decision in 1984Q2 |
| Validation split | 2002Q1–2006Q4 |
| Final development population | 1,082,584 labelled rows; 1,976 represented CERT–failure-date events |
| Purge years | 2001 and 2007 |
| Historical test split | 2008Q1–2014Q4 |
| Historical test population | 205,672 labelled rows; 2,007 positive rows |
| Historical test failures | 505 represented failures in the test calendar; 513 represented events through the complete forecast horizons |

The modelling unit is a bank–decision-quarter. The target is failure in the subsequent four quarters, under the documented risk-set and outcome-adjudication rules. Assistance transactions are not automatically failures. Unconfirmed institutional exits are not automatically classified as mergers. The primary feature policy assumes a 90-day reporting lag and a maximum report age of two quarters. Observations with unresolved outcomes are censored rather than treated as negative labels. Financial amounts in the source are in thousands of USD and reported ratios are in percent.

See the [field contract](docs/field_contract.md), [quality protocol](docs/part2_protocol.md), and [feature protocol](docs/part4_protocol.md) for field definitions, event handling, availability assumptions and split accounting.

## 4. Pipeline architecture and repository structure

```text
FDIC BankFind API
        |
        v
Part 1: Verified Bronze acquisition ----> Storage benchmark evidence
        |
        v
Part 2: Silver financial panel + dated event ledger
        |                         |
        |                         +--> Part 3: Processing benchmark evidence
        v
Part 4: Gold decision-time features + horizon-specific labels
        |
        v
Part 5: Chronological development --> SEALED models and alarm policies
        |
        v
        Historical test --> VERIFIED saved predictions and metrics
        |
        v
Part 6: Read-only synthesis of all upstream evidence
```

The processing experiments use the original Silver panel and synthetic replicas of financial histories. Synthetic records and retrospective terminal classifications do not enter the modelling population.

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/workflows/tests.yml
├── run_part1.py ... run_part6.py
├── docs/
│   ├── field_contract.md
│   └── part1_protocol.md ... part6_protocol.md
├── notebooks/
│   └── 01_acquisition_colab.ipynb ... 08_conclusions_colab.ipynb
├── src/fdic_ews/
│   ├── acquisition.py, api.py, contracts.py
│   ├── cli.py, storage.py, benchmark.py, worker.py
│   ├── quality/
│   ├── processing/
│   ├── features/
│   ├── modelling/
│   └── conclusions.py
└── tests/
    ├── unit/
    └── integration/
```

The CLI entry points run directly from a repository checkout. The notebooks invoke the same research implementation and display its recorded evidence. Research protocols specify the methodological contracts and interpretation boundaries for each stage.

## 5. Environment and automated validation

### Prerequisites

- Linux or Google Colab with a CPU runtime; no GPU, paid cloud or FDIC API key is required.
- Python 3.11, 3.12 or 3.13 and Java 17 or later for PySpark and the full integration test suite.
- Internet access for package installation and FDIC acquisition; sufficient persistent storage and separate local scratch space for benchmark representations.
- A consistent Python and package environment for model development and historical evaluation. The recorded modelling run uses Python 3.13.18.

The 16 direct dependencies are pinned in [requirements.txt](requirements.txt), including PyArrow 21.0.0, pandas 2.2.3, Polars 1.35.2, PySpark 4.0.1, scikit-learn 1.7.2 and LightGBM 4.6.0. The fixed research seed is `11230945`. Pinned direct dependencies do not constitute a complete operating-system or transitive-dependency lock.

[GitHub Actions](https://github.com/NguyenDuong0508/11230945/actions) runs the complete offline test suite on Python 3.11 and 3.12 with Java 17. Tests use synthetic FDIC-shaped fixtures; they do not reacquire live data or rescore the reported historical test. All tests must pass without skips before running the research pipeline. The recorded final suite contains 142 tests. A unit-only run is suitable for debugging, not a substitute for the integration gate.

## 6. Reproduction in Google Colab

Open the notebooks below in order. Use a fresh CPU runtime, install the pinned dependencies through the setup cell, and complete the mandatory offline validation cell. Mount your **own** Google Drive and use the same workspace (`/content/drive/MyDrive/11230945`) throughout. Access to the original researcher's personal Drive is not required. For independent reproduction, ensure this is a dedicated workspace that has not previously been used for a historical test.

| Part | Notebook | Principal output |
|---|---|---|
| 1 | [01 — Acquisition and Source Verification](notebooks/01_acquisition_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/01_acquisition_colab.ipynb) | Verified Bronze source |
| 1 | [02 — Storage Experiments](notebooks/02_storage_experiments_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/02_storage_experiments_colab.ipynb) | Storage equivalence and measured performance |
| 2 | [03 — Data Quality and Panel](notebooks/03_quality_panel_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/03_quality_panel_colab.ipynb) | Verified Silver panel and quality evidence |
| 3 | [04 — Scalable Processing](notebooks/04_scalable_processing_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/04_scalable_processing_colab.ipynb) | Engine parity, plans, timing and memory evidence |
| 4 | [05 — Feature Pipeline](notebooks/05_feature_pipeline_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/05_feature_pipeline_colab.ipynb) | Verified Gold feature and outcome table |
| 5 | [06 — Development and Model Selection](notebooks/06_model_development_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/06_model_development_colab.ipynb) | Sealed development evidence |
| 5 | [07 — Historical Test and Evaluation](notebooks/07_historical_test_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/07_historical_test_colab.ipynb) | Verified historical evaluation |
| 6 | [08 — Conclusions and Evidence Synthesis](notebooks/08_conclusions_colab.ipynb) · [Colab](https://colab.research.google.com/github/NguyenDuong0508/11230945/blob/main/notebooks/08_conclusions_colab.ipynb) | Cross-stage evidence synthesis |

The committed notebooks contain recorded outputs. They can be inspected on GitHub without executing the code. Test counts in earlier notebooks reflect their execution revisions; the complete suite at the final synthesis stage contains 142 tests. Do not overwrite recorded results merely to equalize notebook execution counts or test counts.

For Notebook 01, complete the initial one-quarter acquisition check and resume until all 170 quarters and both reference tables pass verification. Notebook 02 uses local scratch storage for timed reads. In each subsequent stage, resolve the verified upstream pointer before processing. Follow the notebook's instructions to create evidence only if needed; do not regenerate upstream artifacts in a completed workspace.

Before a first evaluation in an independent workspace, inspect the sealed development results in Notebook 06 and the runtime, thresholds and cost policy in Notebook 07. Explicitly set `AUTHORIZE_HISTORICAL_TEST = True` only when ready to execute that workspace's single historical evaluation. Leave the flag at `False` for inspection. If a verified test already exists, Notebook 07 follows the read-only verification branch regardless of the flag. An incomplete attempted test must be preserved and documented, not repeated.

Notebook 08 requires a completed verified evidence chain. It does not acquire data, rerun benchmarks, fit models, optimize thresholds or score the test.

## 7. End-to-end reproduction from a Linux checkout

These commands provide an alternative to Colab. Run the stages in the same Bash session, stopping at any failure. Use a new workspace for independent reconstruction, not the original workspace containing the reported historical test. Record the Git revision used, and keep that checkout and environment unchanged throughout development and evaluation.

### 7.1. Setup and offline test gate

```bash
set -e

git clone https://github.com/NguyenDuong0508/11230945.git
cd 11230945

git rev-parse HEAD
python -m venv .venv
source .venv/bin/activate

python --version
java -version
python -m pip install -r requirements.txt
python -m pip check

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_PYTHON="$(command -v python)"

python -m pytest -q --junitxml=/tmp/bft65b-tests.xml
python -c "import xml.etree.ElementTree as E; s=E.parse('/tmp/bft65b-tests.xml').findall('.//testsuite'); assert s and sum(int(t.get('tests',0)) for t in s)>0 and sum(int(t.get('skipped',0)) for t in s)==0"
```

Expected result for the recorded suite: `142 passed`, zero skips. Resolve dependency, Java or test failures before proceeding.

### 7.2. Dedicated workspace

```bash
export WORKSPACE="$(mktemp -d "$HOME/fdic_ews_11230945.XXXXXX")"
export SCRATCH="$(mktemp -d /tmp/fdic_ews.XXXXXX)"
printf 'Workspace: %s\nScratch: %s\n' "$WORKSPACE" "$SCRATCH"
```

Record these paths. On interruption, resume using the same paths rather than creating a different workspace. Keep scratch and workspace separate from one another and from the Git checkout. Use a single writer. Retain the workspace after completion for evidence inspection.

### 7.3. Parts 1–4: Acquisition, storage, panel, processing and features

```bash
# Part 1 — Acquire and verify all required source data.
python run_part1.py acquire --workspace "$WORKSPACE" --max-new-quarters 1
python run_part1.py acquire --workspace "$WORKSPACE"
python run_part1.py verify --workspace "$WORKSPACE"

# Part 1 — Compare storage representations on local scratch.
python run_part1.py benchmark --workspace "$WORKSPACE" \
    --local "$SCRATCH/storage" --warmups 1 --repetitions 5 --threads 2

# Part 2 — Build and verify the Silver panel.
python run_part2.py build --workspace "$WORKSPACE"
python run_part2.py verify --workspace "$WORKSPACE"

# Part 3 — Run and verify scalable-processing experiments.
python run_part3.py run --workspace "$WORKSPACE" --scratch "$SCRATCH/processing"
python run_part3.py verify --workspace "$WORKSPACE"

# Part 4 — Build and verify the Gold feature table.
python run_part4.py build --workspace "$WORKSPACE"
python run_part4.py verify --workspace "$WORKSPACE"
```

A partial acquisition is not an accepted Bronze run. Acquisition resumes pending work or verifies a sealed source. The optional `acquire --fresh` initiates a different source vintage; it is not required for the normal sequence above and must not be used to replace the completed research snapshot. A failed benchmark may retain scratch for diagnosis. Do not modify sealed artifacts or switch active pointers in a completed modelling workspace.

### 7.4. Part 5: Development only

```bash
python run_part5.py develop --workspace "$WORKSPACE"
```

Stop here and inspect the printed development summary and the `development_manifest.json` under `modelling/<Gold manifest SHA-256>/`. Confirm `SEALED` state, selected model, feature specification, cost policy, thresholds and the recorded runtime. The CLI does not automatically open the historical test as part of development.

### 7.5. Part 5: Explicit historical evaluation

Only for an independent reproduction workspace with a sealed development run and no previous test attempt, execute the following **once** after reviewing the development evidence:

```bash
python run_part5.py evaluate --workspace "$WORKSPACE"
```

Never use `evaluate` to inspect an already completed test. If the attempt fails after its marker is created, preserve the marker and artifacts; investigate and document the failure rather than deleting the marker or repeating the test. For a completed evaluation, use:

```bash
python run_part5.py verify --workspace "$WORKSPACE"
```

The verification command recomputes metrics from saved prediction evidence without loading fitted estimators or generating new scores.

### 7.6. Part 6: Read-only synthesis

```bash
python run_part6.py --workspace "$WORKSPACE"
```

This command prints the aggregate evidence as JSON. It verifies the upstream identities, file hashes, population accounting and economic arithmetic, and creates no new workspace artifacts. See Notebook 08 for the presentation of the results.

For an existing completed workspace, use only the Part 5 `verify` and Part 6 synthesis commands for model-evidence inspection. Do not repeat the acquisition, benchmark, feature-build or development steps merely to inspect the reported evaluation.

## 8. Research design and stage contracts

### Part 1 — Acquisition and Storage

The Bronze layer uses typed Snappy Parquet partitioned by calendar year and quarter, with separate reference tables. The acquisition checks all expected quarters, API-reported totals, required keys, schemas and physical read-back results. Numeric blanks become null; invalid required keys or unparsable numeric fields stop acquisition. Reported zeros and negative values are retained.

Five representations (CSV, flat Snappy, flat ZSTD, partitioned Snappy, partitioned ZSTD) are compared using equivalent full scan, six-column projection, annual and quarterly workloads. A separate pandas–Polars CSV projection comparison is included. Full round-trip equivalence and workload parity precede timing. The default design uses 22 warm-ups and 110 measured trials in fresh worker processes, randomized with the fixed seed. Library import and startup are outside timing; OS page caches are not flushed. Partition candidate counts indicate pruning opportunities, not bytes transferred. Flat ZSTD minimizes the measured footprint; partitioned Snappy supports incremental acquisition and efficient temporal access. [Protocol](docs/part1_protocol.md)

### Part 2 — Data Quality and Panel

The Silver layer retains financial records and documents missingness, reporting gaps, capital zeros and financial anomalies. A dated event ledger separates failure, assistance and uncertain institutional ends. Retrospective terminal audits remain separate from historical predictors and are not used as blanket sample exclusions. Financial anomalies are not automatically errors; treatment is field-specific. Reporting gaps are not filled. [Protocol](docs/part2_protocol.md)

### Part 3 — Scalable Processing

Native pandas, Polars and local PySpark implementations are compared on equivalent filtered aggregation, adjacent-quarter growth and state–quarter peer-join workloads. The controlled eight-column projection is tested at 1×, 5× and 10× using disjoint synthetic CERT histories for the larger scales. Spark is evaluated at 1× only; one- and two-thread Polars scenarios assess parallelism. Output parity, optimized/physical plans, median latency and sampled process-tree RSS are retained. The results support single-node Polars for the evaluated workloads; local Spark is not evidence of a multi-node cluster advantage. [Protocol](docs/part3_protocol.md)

### Part 4 — Feature Pipeline

The Gold layer uses the latest report available under the explicit 90-day lag and two-quarter age cap, exact-quarter transformations and horizon-specific labels. Its 29 raw predictors cover profitability, capital, asset quality, funding, growth, peer deviation and reporting quality. The fixed chronological split includes four-quarter purges. Source event fields, bank identifiers, failure costs and retrospective terminal classifications do not enter the predictor matrix. Median imputation, missingness indicators and robust scaling are fitted on training observations only. The capital-zero masking alternative is a sensitivity analysis rather than a blanket correction. [Protocol](docs/part4_protocol.md)

### Part 5 — Modelling and Validation

Four purged expanding-window folds compare a training-prevalence prior, Logistic Regression, Histogram Gradient Boosting and five LightGBM specifications. Fresh preprocessing is fitted inside each fold. The primary estimators retain natural training prevalence. Equal-weight mean fold Average Precision (AP) determines selection; pooled out-of-fold scores are complementary diagnostics. The selected `lightgbm_no_quality` specification retains 22 raw predictors and excludes seven predefined reporting-history and quality predictors; preprocessing can still produce missingness indicators.

The decision loss charges each false-alarm bank-quarter observation and each distinct missed failure event once. No positive FDIC cost record satisfies the historical cost-vintage screen for development. The predeclared fallback uses 0.1% of median development assets for false alarms (USD 60,210) and 10% of median failed-event assets for missed events (USD 4,422,500). These are data-scaled assumptions, not observed costs. The selected threshold is `0.03367313153323793`; the prior independently selects `2.0`, a no-alarm sentinel. Both policies are frozen before evaluation.

A separate Spark MLlib demonstration uses five financial predictors, subsampled training negatives and full natural-prevalence F1 validation; it does not participate in model selection. The final selected model and prior are refitted on eligible development data, sealed and evaluated on the fixed historical test. No test-time fitting, tuning or threshold adjustment is performed. [Protocol](docs/part5_protocol.md)

### Part 6 — Conclusions and Evidence Synthesis

The final stage checks the Bronze, storage, Silver, processing, Gold, development and test evidence identities. It reconciles source and split populations, benchmark summaries, annual additive row counts and fixed-policy economic loss. Distinct failure events are not additive across decision years. The synthesis answers Q1–Q3 and specifies limitations and local adaptation requirements for Vietnam. It is an artifact-integrity and aggregate-arithmetic check, not a new independent evaluation. [Protocol](docs/part6_protocol.md)

## 9. Workspace, evidence integrity and research limitations

Generated artifacts are maintained outside Git. The workspace structure includes:

```text
workspace/
├── active_source.json
├── bronze/<run-id>/
│   ├── manifest.json
│   ├── verification.json
│   ├── financials/year=YYYY/quarter=Q/part.parquet
│   ├── failures.parquet
│   └── institutions.parquet
├── active_storage.json
├── storage_evidence/<run-id>/
├── active_panel.json
├── silver/<run-id>/
├── active_processing.json
├── processing_evidence/<run-id>/
├── active_features.json
├── gold/<run-id>/
└── modelling/
    ├── test_attempt.json
    └── <Gold manifest SHA-256>/
        ├── development_manifest.json
        ├── saved development evidence and fitted estimators
        └── test/
            ├── manifest.json
            └── saved predictions and evaluation evidence
```

`pending_source.json` is present only during a pending acquisition. Active pointers identify accepted runs and their manifest digests. The workspace-level test-attempt marker is created exclusively before test partitions are decoded. Never remove it, alter the sealed models or change upstream identities to repeat evaluation. Integrity checks detect accidental artifact changes; they are not a substitute for trusted storage or independent validation.

The following SHA-256 identities are recorded in Notebook 08, Table 2, for the reported experiments. They identify that research run; an independent acquisition or reconstruction is not expected to reproduce these exact digests.

| Evidence | Recorded SHA-256 |
|---|---|
| Bronze | `e376152baac058e4dd4fff4582e9afa3c58820fff2f278115a82bff7fd4b3f6e` |
| Storage | `8516873cff244b6f943d2c282753093ef6ea2b6c64eba107b82decf951d9dc78` |
| Silver | `b6d91c7d0095b81e6200a0ae0940a103968c452046b168e446cd02089d323af0` |
| Processing | `36ca1cbdd57b4fa74abc67fac85dafdb73d0080b1f73ea8f2aa086fbdeaf063f` |
| Gold | `de2abf748e29025760f50cc6c3c57d899a4402fe079756e17f9a6d93f21b5780` |
| Development | `12532646188770d748eea1b494cb350410a23057887e552ba5568ec19c04072f` |
| Historical test | `8e6d5344d8a0bbb7fa56ff50be8f7b53796f89d6367cf8dc011ad72fae6298cd` |

### Reproduction boundaries

- **Source vintage:** The public FDIC API is live, not a permanently versioned research archive. Reacquisition can differ in values, counts and checksums. Matching API totals do not prove transactional snapshot consistency. Exact verification of the recorded results requires the original artifacts, which are retained outside the public repository and are not distributed here. The public code and notebooks support independent reconstruction and inspection of recorded outputs.
- **Historical availability:** Actual historical filing and revision timestamps are unavailable. The publication lag is an assumption; it does not establish complete historical vintage correctness. Uncertain exits and censoring can restrict the evaluation population.
- **Evaluation:** The 2008–2014 period is reserved as a chronological out-of-time holdout. All feature, model and alarm-policy choices are determined exclusively from development data and sealed before the historical evaluation. The recorded protocol permits a single test evaluation, followed only by read-only verification of saved predictions and metrics. The retrospective design does not establish prospective or 2023-crisis performance.
- **Decision costs and calibration:** Asset-based costs are scenario assumptions. The threshold is transferred from out-of-fold scores to the final refitted model. Event recall for 2008 decision observations is 47.24%, demonstrating operating instability despite strong aggregate ranking. Lower Brier score does not establish perfect calibration.
- **Computational scope:** Timings and memory depend on hardware, software, caching and workload. Synthetic scale-up does not represent all properties of genuine future histories. Local Spark cannot establish multi-node break-even or cluster economics.
- **Vietnam adaptation:** Local reporting timestamps, adjudicated events, accounting definitions, cost and capacity estimates, independent validation, human review and governance are required before operational use.

## 10. Sources and references

### Data and course materials

- Federal Deposit Insurance Corporation (FDIC). [BankFind Suite API documentation](https://api.fdic.gov/banks/docs/); [financial field definitions](https://api.fdic.gov/banks/docs/risview_properties.yaml); [failure field definitions](https://api.fdic.gov/banks/docs/failure_properties.yaml); [institution field definitions](https://api.fdic.gov/banks/docs/institution_properties.yaml); [bulk data](https://banks.data.fdic.gov/bankfind-suite/bulkData).
- FDIC. [Failures and Assistance definitions](https://banks.data.fdic.gov/bankfind-suite/failures/help); [Disclaimers and Methodologies](https://banks.data.fdic.gov/bankfind-suite/help?helpTopic=disclaimers-and-methodologies); [Community Banking Research Program](https://www.fdic.gov/community-banking-research-program).
- National Economics University. *BFT65B — Big Data Analytics in Finance*, final individual research report instructions, course lecture materials and `fdic_download_starter.py`. The starter provides the field-selection and quarter-by-quarter retrieval starting point.

### Computational methods and software

- Apache Arrow. [CSV](https://arrow.apache.org/docs/python/csv.html), [Parquet](https://arrow.apache.org/docs/python/parquet.html) and [Dataset](https://arrow.apache.org/docs/python/dataset.html) documentation.
- pandas developers. [pandas documentation](https://pandas.pydata.org/docs/); Polars developers. [Polars documentation](https://docs.pola.rs/) and [lazy query optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/).
- Apache Software Foundation. Spark 4.0.1: [SQL performance tuning](https://spark.apache.org/docs/4.0.1/sql-performance-tuning.html), [PySpark installation](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html), and [ML pipelines](https://spark.apache.org/docs/4.0.1/ml-pipeline.html).
- scikit-learn developers. Version 1.7 documentation: [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html) and [Model evaluation](https://scikit-learn.org/1.7/modules/model_evaluation.html).
- LightGBM developers. [Parameters, version 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
- Supporting libraries: [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/), [Requests](https://requests.readthedocs.io/), [Matplotlib](https://matplotlib.org/stable/), [psutil](https://psutil.readthedocs.io/), [pytest](https://docs.pytest.org/), [nbformat](https://nbformat.readthedocs.io/), [joblib](https://joblib.readthedocs.io/) and [threadpoolctl](https://github.com/joblib/threadpoolctl).

### Supervisory adaptation

- Basel Committee on Banking Supervision (2013). [Principles for effective risk data aggregation and risk reporting](https://www.bis.org/publications/201301-guidelines-principles-effective-risk-data-aggregation-and-risk-reporting). Bank for International Settlements.
- Basel Committee on Banking Supervision (2024). [Core Principles for effective banking supervision](https://www.bis.org/committees/bcbs/basel-framework/standard/bcp?allChapters=true). Bank for International Settlements.
- National Assembly of Vietnam (2024). [Law on Credit Institutions No. 32/2024/QH15](https://vanban.chinhphu.vn/?classid=1&docid=211190&pageid=27160&typegroupid=3).
- National Assembly of Vietnam (2025). [Law No. 96/2025/QH15 amending and supplementing certain articles of the Law on Credit Institutions](https://vanban.chinhphu.vn/?classid=1&docid=214588&pageid=27160&typegroupid=3).

The supervisory references inform proposed governance and event-taxonomy requirements. They are not an exhaustive statement of applicable law or evidence of model validity in Vietnam. Further methodological sources and implementation details appear in the stage protocols linked above.

## 11. AI-assisted development disclosure

[OpenAI ChatGPT](https://chatgpt.com/) was used to assist with research design, Python implementation, debugging, automated test construction, documentation, experimental visualization and report preparation. Assistance covered acquisition and storage, financial data quality, scalable processing, temporal feature engineering, chronological model validation, cost-sensitive event evaluation and cross-stage evidence verification.

Empirical findings are reported from recorded pipeline executions. Source code, saved notebook outputs, automated tests and artifact manifests document the procedures and evidence. The submitting researcher is responsible for reviewing the implementation, verifying the results and ensuring accurate interpretation and citation in the final report.

Raw data, generated datasets, fitted models, predictions and scratch artifacts must not be committed to this repository. The public GitHub repository and the documented FDIC acquisition procedure are the primary submission and reproduction resources; a personal Google Drive link is not required.
