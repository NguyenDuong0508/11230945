# Part 4 — Feature Pipeline

## Objective and verified inputs

The Gold layer provides a bank–decision-quarter table suitable for a four-quarter failure early-warning model. It consumes the verified Silver financial panel and the dated event ledger, not the Part 3 benchmark replicas or retrospective terminal audit. The Bronze and Silver manifest digests are recorded in the Gold manifest. `active_features.json` is published only after verification, and all data and fitted artifacts remain in the workspace.

Financials are processed with a narrow column projection and a bounded cache of quarterly partitions. Vectorized pandas/NumPy operations and exact-key historical joins keep intermediate financial tables small. Part 3 establishes the case for Polars on large analytical workloads; here the quarter-bounded feature operations and the scikit-learn interface motivate this implementation. No new engine performance claim is made.

## Information set and target

A calendar quarter end `q` is a decision date, not necessarily the date of the latest available report. A report dated `r` is eligible when `r + 90 calendar days <= q` and its age is at most two calendar quarters. For each CERT choose the latest eligible report, using the second previous quarter only when the first is unavailable. The age cap is an explicit operational assumption, not an FDIC reporting rule. The first decision quarter has no pre-study report. A 120-day candidate-coverage comparison retains the same age cap and is restricted to development. The reporting dates and assumed availability timestamps are verified before feature construction.

The as-of risk set excludes dated failure and institutional-end evidence occurring on or before `q`. A report after such an event is separately counted as a prior-exit conflict. Assistance alone is not an exit. No future report, eventual terminal class, or eventual institutional status determines predictor eligibility. Date-of-event availability is assumed for dated institutional evidence; actual publication histories and historical report vintages are not supplied. This is a pseudo-real-time reconstruction under explicit assumptions, not verified real-time data availability.

For each at-risk CERT, the target is 1 for an adjudicated failure in `(q, q+4 quarters]`, and 0 when a verified non-failure exit occurs within the horizon or a report at the horizon establishes continued reporting and there is no contradictory event. Candidate institutional ends without confirmed exit type, incomplete follow-up, unknown resolutions, multiple in-horizon dates and inconsistent in-horizon report/event ordering are censored (`null`), not coded as negatives or positives. Candidate ends after a failure do not invalidate that failure unless other conflicting evidence exists. A report strictly after a failure date within the same outcome horizon is a conflict. Duplicate CERT/date/type evidence is collapsed. Year-9999 open-ended dates are ignored. Undated or unlinked records cannot adjudicate individual horizons and remain a source-linkage limitation. No event or report after the horizon is used to change the label. The ledger does not establish that all dated institutional ends are mergers; the verified-nonfailure branch is supported by the policy but no unconfirmed source end is upgraded to that type. A development-only sensitivity counts additional negatives under the explicit assumption that uncontradicted candidate ends represent non-failure exits.

The decision scope is 1984Q1–2025Q2 (166 quarters), with outcome maturity through 2026Q2. All at-risk rows, including censored rows and rows in unused periods, remain in Gold with a reason. Only non-null labels are eligible for supervised training or evaluation. Risk candidates, prior exits, risk rows, positive and negative labels, and censoring are reconciled by decision quarter and split. Label counts and distinct represented CERT/date events are not interchangeable. A fixed holdout must contain at least 50 distinct represented failure CERTs whose event dates fall within its own calendar period. Report linked-event coverage and the difference between event counts and positive rows; no automatic split changes are allowed.

## Financial features

The 29-feature allowlist in `financial.py` defines the model interface. The dictionary exported by the build gives every feature's group, formula and economic rationale. Size, equity and tangible equity, liquidity, deposit and loan funding, asset quality, reserves, profitability, capital, changes, report quality and relative profitability are covered. All amounts share the source unit of USD thousands. Ratios multiplied by 100 are expressed in percent; ROA changes and peer deviations are in percentage points. Natural log size uses `log1p` of positive assets. The Texas-like indicator is a proxy: `(NCLNLS + ORE)/(EQ - INTAN + LNATRES) × 100` only when the denominator is positive and the distressed-asset, intangible and reserve components are nonnegative and observed. Missing components are not assumed zero. Its historical comparability depends on reporting definitions; it is not a supervisory rating.

For growth and annual change, only exact report-quarter lags `r-1` and `r-4` are used. A missing predecessor or invalid denominator returns null. No forward filling, interpolation or future aggregation occurs. The reported capital fields are retained in the primary specification with separate zero flags. The alternative zero-as-missing policy is fixed before model evaluation, changes the two capital ratio predictors only, and retains their zero flags. It does not establish which reported zeros are non-applicable. FDIC income and expense reporting conventions have changed over the study; the primary feature set uses reported ROA, ROE and NIM and excludes conditional quarterly-income candidates from Part 2 rather than asserting historical YTD harmonization.

ROA peer deviation is calculated before outcomes are joined: the bank's ROA minus the leave-one-out mean of other available at-risk institutions in the same state and source reporting quarter, with non-null ROA and nonempty state. A singleton or missing peer is null. The group may include institutions whose eventual label is censored; outcomes do not select peer membership. Peer features are cross-sectional as-of transformations rather than fitted preprocessing statistics. CERT, dates, split, label, reason, event evidence and cost are not model predictors.

## Fixed chronological design

| Decision quarters | Role |
|---|---|
| 1984Q1–2000Q4 | Training (first quarter may be empty) |
| 2001Q1–2001Q4 | Purge |
| 2002Q1–2006Q4 | Validation |
| 2007Q1–2007Q4 | Development–test purge |
| 2008Q1–2014Q4 | Historical test |
| 2015Q1–2025Q2 | Unused post-test audit |

The early development period contains the banking stress of the late 1980s and early 1990s; the test covers the later crisis and recovery. Four-quarter purges separate the final development horizons from the next evaluation period. The 2008–2014 test period is reserved as a chronological out-of-time holdout. No test-performance evidence is used to guide feature, model or threshold choices. Part 4 checks split and label counts only; it does not score models or fit preprocessing on holdout data. The post-test panel is not an additional tuning set. The retrospective design does not establish prospective performance.

## Training-only preprocessing and handoff

`predictor_matrix` selects only the fixed ordered 29 features and optionally masks the two reported capital-zero fields. A scikit-learn `Pipeline` fits `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)` and `RobustScaler(quantile_range=(25,75))` on labelled training rows only. All-null features are retained with the imputer's empty-column handling; constant and empty training features are reported, not selected using the holdout. The learned median, center, scale, missingness indicators and transformed names are exported. Training-only transformation checks require finite outputs. The primary and capital sensitivity pipelines are fitted independently on exactly the same labelled training population. No automatic selection of the preferable capital treatment is performed.

Part 5 must construct a fresh preprocessing pipeline inside each chronological training fold, apply it to that fold's validation rows, choose features/model/threshold only on development, then refit the locked estimator and preprocessing on eligible development data before a single test evaluation. The saved Part 4 estimators document the initial training-only fit and are not a replacement for fold-specific fitting. Do not load untrusted joblib files. Use the raw Gold predictors and the explicit allowlist; keep cost and event identifiers outside the estimator. Cost is event-level evidence, not a per-bank-quarter feature.

## Verification and reproducibility

The build validates unique CERT–decision keys, exact dates and horizons, binary/null labels, event intervals, finite-or-null predictors, split membership and quarterly accounting. Each Parquet partition is read back, then all outputs are recomputed from Silver and compared with saved tables, audit CSVs and training-only preprocessing diagnostics. Hashes protect fitted artifacts; numerical diagnostics are independently recomputed, not recovered from serialized estimators. The run fails if both classes are not represented in each supervised split or if the fixed holdout has fewer than 50 represented failure institutions in the test calendar. Unit tests exercise exact-quarter lags, future-event invariance, early failures, assistance, conflicting events, competing exits, peer membership, train-only fitting and split boundaries. A full-calendar synthetic integration test exercises Bronze-to-Gold Parquet and the holdout gate.

Run `python run_part4.py build --workspace /path/to/workspace` and `python run_part4.py verify --workspace /path/to/workspace`. The evidence includes `panel/`, `manifest.json`, `summary.json`, `quarter_audit.csv`, `split_summary.csv`, `label_reasons.csv`, `feature_dictionary.csv`, `training_feature_quality.csv`, `preprocessing.json` and two fitted preprocessor artifacts. See Notebook 05 for presentation and diagnostics. Do not commit any workspace data or serialized estimators.

## Sources

- FDIC [BankFind Suite API and field definitions](https://api.fdic.gov/banks/docs/), [Failures and Assistance](https://banks.data.fdic.gov/bankfind-suite/failures/help), [Disclaimers and Methodologies](https://banks.data.fdic.gov/bankfind-suite/help?helpTopic=disclaimers-and-methodologies).
- BFT65B final examination instructions and course materials on data transformation, pipelines, feature selection and time-series validation.
- scikit-learn [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).
