# Part 6 — Conclusions and evidence synthesis

## Purpose

Answer the three BFT65B research questions using the completed evidence chain. Q1 concerns laptop storage and processing and conditions for a Spark cluster; Q2 concerns data quality and look-ahead safeguards; Q3 concerns honest early-warning evaluation. Interpret the results under the measured scope, identify limitations, and specify changes required for a Vietnamese supervisory deployment. This stage does not train, select, score, calibrate or optimize a model.

## Read-only evidence contract

Use the same workspace as Parts 1–5. `run_part6.py --workspace PATH` returns a JSON summary to standard output. Notebook 08 renders the same summary. No additional workspace pointer, dataset, model or persistent artifact is created. The active Bronze, storage, Silver, processing and Gold pointers must resolve inside their expected directories; all stage identities must match the ancestry of the sealed modelling run. Verify the file hashes against the manifests, including the Bronze Parquet inventory and the workspace test-attempt marker. Reconcile Bronze counts with the saved verification report, Silver counts with Bronze, Gold split accounting, test population and event counts with Gold, and annual additive row metrics and the economic loss with the saved operating evidence. Distinct events are not additive across decision years. All evidence is inspected after the test is closed; no new scoring or threshold selection is permitted.

The synthesis is a byte-integrity and aggregate-arithmetic check, not a full reconstruction of raw-data transformations, a replay of the benchmark, or a new independent prediction validation. The earlier notebooks contain the full verification evidence. Part 5's `verify` command may be used for read-only recomputation from saved predictions; it must never be replaced by `evaluate` during synthesis. Hashes do not prove that a deliberately rewritten manifest is authentic. Use a single writer throughout upstream stages; do not change active pointers while reading the evidence. If a stage is missing, corrupt, mismatched or unsealed, stop and document the issue rather than fabricate or substitute values.

## Q1 — Storage and scalable execution

Compare all five storage representations against CSV by size and median workload latency, using equivalent typed outputs. Include partition candidate counts for the year and quarter predicates; these are not bytes read. Compare pandas and Polars at 1×, 5× and 10× on the original and synthetic replicated histories; compare Spark only at 1×. Derive workload-specific speedups, within-engine scale ratios and observed process RSS. Use recorded RAM as context, not a capacity guarantee. Use the parallelism table and optimized/physical plans in Notebook 04 to explain the results. Local Spark is not a multi-node cluster benchmark, and synthetic replication is not a forecast of genuine future institutional histories. Retain the verified Bronze storage layout; a recommendation about another codec or layout does not rewrite it. A cluster decision depends on measured memory pressure, latency targets, concurrency, data locality, network overhead and total cost; a quantitative break-even requires a separate multi-node experiment.

## Q2 — Quality and temporal integrity

Report missingness and zeros by field, anomaly flags, reporting gaps, event taxonomy and retained rows from Silver. A flagged financial ratio need not be an error. Keep failure, assistance, unconfirmed institutional end and unresolved discontinuation distinct. Show the Gold risk-set, labelled, positive and censored counts separately; compare the 90-day primary lag with source-report availability sensitivity and the fixed horizon-invariance case. Actual filing and revision timestamps are unavailable; a lag assumption is not proof of historical vintage correctness. Conditional YTD differences and capital-zero masking are sensitivity evidence rather than blanket accounting corrections. Do not treat terminal audit classes as historical predictors or as global sample exclusions. Use horizon-specific outcome adjudication, exact-quarter lags and train-only preprocessing. Censored rows restrict the population to adjudicable outcomes.

## Q3 — Historical early-warning feasibility

Report the selected specification and locked threshold, AP, ROC-AUC, probability error, top-5% quarter-wise row recall, alert rate and precision. Separate positive bank-quarter recall from distinct CERT–failure-date event recall and failures inside the test calendar. Report the fraction of all represented events detected at least 90 days ahead, not only lead time conditional on detection. Compare the selected policy against independently thresholded prior and no-/all-alarm references under the sealed cost assumptions. The asset-scaled cost scenario is not observed supervisory expenditure or realized savings. Show annual decision-period diagnostics and identify periods with weak operating recall even if ranking remains strong. No model or threshold is revised from these diagnostics. The historical holdout was explored in prior research and is retrospective, not an independently blind prospective sample. No claim of calibration, causal prevention, deployment readiness or performance on the 2023 episode follows from this test.

## Application to Vietnam — requirements, not findings

| Layer | Required adaptation | Acceptance evidence |
|---|---|---|
| Acquisition | Authorized supervisory submissions, stable institution and consolidation-scope identifiers, reporting and revision timestamps, versioned schemas, secure access | Reconciled submission inventory, versioned filing history, access audit |
| Quality and events | Local accounting definitions, currency and units, regulatory regime changes, dated mergers, restructuring, intervention, special control, dissolution and failure; legal/event adjudication by authorized experts | Validated mapping dictionary, independently reviewed event ledger, explicit competing outcomes |
| Features | Locally consistent capital adequacy, asset quality, provisioning, profitability, liquidity, credit concentration and funding measures; actual availability times | As-of feature audit and local reporting-regime sensitivity |
| Validation | Locally defined outcomes and forecast horizon, chronological development, purges and independently held-out or prospective evaluation; quantify sparse events and selection | Predeclared local protocol, event counts, calibration and uncertainty diagnostics |
| Decisions | Estimate review costs, capacity and missed-event severity from authorized local evidence; choose threshold on development only | Capacity-aware policy, independent validation, human review and escalation |
| Governance | Role-based access, encryption, logging, model/version approval, drift monitoring, supervisory accountability | Controlled pilot, audit trails and documented approval criteria |

US feature values, fitted estimators, thresholds, estimated losses and test metrics are not directly transferable. The adaptation matrix is a proposed engineering and validation plan, not an assessment of Vietnamese bank risk. Confirm applicable legal provisions and reporting requirements with authorized local experts before specifying labels or production access. BCBS 239 provides a reference for risk-data governance; it is not asserted to be a universal local legal obligation.

## References

- National Economics University, BFT65B final examination instructions, research questions Q1–Q3 and grading rubric.
- Basel Committee on Banking Supervision (2013), [Principles for effective risk data aggregation and risk reporting](https://www.bis.org/publications/201301-guidelines-principles-effective-risk-data-aggregation-and-risk-reporting).
- Basel Committee on Banking Supervision (2024), [Core Principles for effective banking supervision](https://www.bis.org/committees/bcbs/basel-framework/standard/bcp?allChapters=true).
- National Assembly of Vietnam, [Law on Credit Institutions No. 32/2024/QH15](https://vanban.chinhphu.vn/?classid=1&docid=211190&pageid=27160&typegroupid=3) and [amending Law No. 96/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214588&pageid=27160&typegroupid=3). These are reference instruments, not an exhaustive statement of the law in force. Verify subsequent amendments before deployment.
- Parts 1–5 protocols, executed notebooks and sealed workspace evidence.
