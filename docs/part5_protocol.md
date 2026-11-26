# Part 5 — Modelling and Validation

## Objective and population

The verified Gold table supplies 29 raw predictors, decision-date keys, mature four-quarter labels and split metadata. The modelling unit is a labelled bank–decision-quarter. Censored rows, both designated purge periods (2001 and 2007) and all post-test decisions are excluded from supervised fitting and evaluation. No source event field, bank identifier, split identifier, outcome metadata, synthetic benchmark record or resolution cost enters the predictor matrix. Gold and Silver inventories and all three upstream manifest identities are checked. This is a byte-integrity/lineage check of the previously verified artifacts, not a second reconstruction of Gold. The original Bronze manifest is checked for identity; its raw payload is not reread by modelling. The development reader decodes only training and validation Gold partitions, not test partitions. The chronological test is fixed at 2008Q1–2014Q4 and reserved for a single out-of-time evaluation after all development choices have been sealed. The test uses historical data and does not establish prospective performance.

## Chronological validation

The development window is the union of the labelled 1984Q1–2000Q4 and 2002Q1–2006Q4 samples. The later validation regime contains relatively few failures. Four nonoverlapping validation blocks, separated by four-quarter gaps, provide a broader development comparison. The final training outcome horizon must end before the validation block starts. The same CERT may appear in training and validation at different dates, reflecting repeated supervision of existing institutions rather than generalization exclusively to unseen banks.

| Fold | Training through | Purge | Validation decision dates |
|---|---|---|---|
| F1 | 1987Q4 | 1988 | 1989Q1–1992Q4 |
| F2 | 1992Q4 | 1993 | 1994Q1–1997Q4 |
| F3 | 1997Q4 | 1998 | 1999Q1–2000Q4 |
| F4 | 2000Q4 | 2001 | 2002Q1–2006Q4 |

All training windows begin at the first available Gold decision. Folds may reuse earlier validation observations only after their outcome horizons have matured. The final fit uses both eligible development samples, not their intervening purge. Fold gaps are relative to each fold and do not permanently remove those observations from other permissible training windows. Every fold must contain both classes in its training and validation populations; failure is a stop condition, not a reason to silently drop a fold. No random row CV, random validation for early stopping, holdout hyperparameter search or test-informed calibration is performed.

## Model specifications

Eight predeclared specifications share the same fold populations. A training-prevalence DummyClassifier is the no-information baseline. Logistic regression uses L2 regularization (C=0.5), with deterministic clipping of robust-scaled values to [-25,25] to limit extreme numerical leverage. Histogram Gradient Boosting uses 80 iterations, learning rate 0.07, 15 leaves and minimum 120 observations per leaf; random early stopping is disabled. LightGBM uses 120 trees, learning rate 0.05, 15 leaves, maximum depth 5, minimum 120 observations per leaf and L2 penalty 1. LightGBM runs with two threads, deterministic mode and column-wise histogram construction. The fixed seed is 11230945. Neither class weights nor resampling are used for these primary estimators. No claim is made that the resulting scores are calibrated probabilities. The baseline uses training prevalence directly; each nonbaseline fits a fresh median imputer (with training-derived indicators), RobustScaler and classifier inside each fold. All learned transformations are fold-specific, and training-only zero masking is a fixed transformation applied consistently before preprocessing.

| Specification | Raw predictors | Treatment |
|---|---:|---|
| prior | 29 (ignored by estimator) | Training prevalence |
| logistic | 29 | Primary values |
| hist_gradient | 29 | Primary values |
| lightgbm_full | 29 | Primary values |
| lightgbm_core | 21 | Exclude tangible equity, loans/deposits, allowance/loans, Texas-like ratio, both capital-zero flags, report age, peer deviation |
| lightgbm_no_peer | 28 | Exclude peer deviation |
| lightgbm_no_quality | 22 | Exclude three reporting-history and four data-quality predictors |
| lightgbm_zero_missing | 29 | Mask two reported capital zeros; retain zero flags |

The four LightGBM alternatives are controlled ablations or treatment sensitivity checks relative to full LightGBM. Mean fold average precision (equal weight per block) selects the highest-ranked nonbaseline specification, with deterministic declaration-order tie breaking. Report the baseline even if no candidate outperforms it. Pooled OOF AP, early-fold mean AP and late-fold AP diagnose heterogeneity; pooled predictions come from different fitted models and may have different score scales. The candidate family and settings are fixed before test; the fold scores used for selection are not independent post-selection estimates. No claim of optimal hyperparameters is made. Feature gain or standardized coefficient magnitude, if available, is a descriptive final-development-fit diagnostic and not causal or held-out importance. The full versus ablated fold comparisons remain the principal contribution analysis.

## Ranking and operational measures

Average precision (AP) is primary because failure is rare. ROC-AUC, Brier score and log loss complement AP; calibration bins report mean predicted score against observed rate without fitting a calibrator. Recall at the top 5% is calculated within each decision quarter, rounding up the capacity and breaking ties by CERT, then pooling positive hits. This ranking metric is measured on adjudicable labelled rows, not the entire operational risk set. The binary operating rule is score >= threshold. Report TP, FP, FN, TN, precision, recall, false-positive rate and alert rate. At the event level, the unit is a unique CERT–failure-date pair: an event is detected if any of its positive decisions is alerted. Lead time is calendar days from its earliest correct alert to failure, calculated only for detected events. Counts within the test calendar are separated from all events represented through the final test horizon. Annual decision-period event counts and losses are not additive because a single event can appear in multiple years.

## Economic loss and threshold

All monetary values are in nominal USD thousands. FDIC COST denotes an estimated resolution expense, not a demonstrated avoided loss from supervision. The policy uses only development-represented failure events, whose dates are no later than 2007-12-31. A positive COST must be associated with a parseable COSTMOSTRECENTASOF between the event date and this cutoff. Deduplicate exact CERT/date costs; conflicting eligible cost values are excluded. Require at least 20 distinct eligible events to estimate a median resolution-cost proxy. The date field is a conservative screen, not proof of a fully verified historical cost vintage. Report the number of eligible events and the number of development events with a positive current cost separately. Never use a later-vintage cost to set the primary threshold. If the eligible sample is insufficient, use the predeclared asset-based missed-event scenario: 10% of median assets at the latest positive development decision per distinct event. The asset amount is recovered from LOG_ASSET. This fallback is an assumed severity fraction applied to data, not a measured FDIC expense. The false-alarm review cost is an explicit assumption of 0.1% of median assets across labelled development observations; actual supervisory review expenditure is unavailable. These assumptions and the fallback trigger are fixed in advance. Report which branch is used, not merely its numerical cost. No inflation adjustment or causal interpretation is claimed.

For threshold t, L(t) = C_FP × number of negative bank-quarter alerts + C_FN × number of distinct represented failure events with no correct alert. The maximum OOF score for each event determines whether that event is missed. Thus multiple positive quarters do not multiply the missed-event cost. No cost is charged for correct alarms in this particular simplified loss; repeated false alarms on negative quarters do incur repeated review costs. Costs of censored observations, intervention, capacity, delayed but successful detection and false alerts with unknown outcomes are not included. Enumerate all distinct OOF scores and a no-alarm sentinel t=2 (all scores in [0,1]); ties choose the largest threshold. The threshold may legitimately select no alarm. Estimate the chosen model and baseline thresholds independently under the same loss. Vary the false-alarm and missed-event unit costs separately by 0.5 and 2 on OOF predictions only. Each scenario has its own units and loss; do not compare scenario losses as if the costs were identical. Threshold selection on OOF scores is empirical; refitting can change score calibration and operational behaviour.

## Spark MLlib demonstration

Use the first chronological fold with five fixed financial features. Sample all positive training observations and up to 20,000 training negatives with the fixed seed. Fit a native Spark median Imputer, VectorAssembler and regularized LogisticRegression in local[2] mode. Stage the small training projection and the complete first-fold validation projection to local Parquet. Evaluate the entire validation block at its natural prevalence. The training sample shifts the fitted prior, and the Spark feature set differs from the primary models; metrics are descriptive and excluded from selection. No distributed-cluster or probability-calibration claim follows from this demonstration.

## Sealing and single evaluation

`develop` writes OOF predictions, fold metrics, model comparisons, cost policy, threshold sensitivity, MLlib evidence and final fitted development estimators for the selected specification and prior baseline. It seals the file inventory, code fingerprint, environment, source identities, selected feature specification and locked thresholds. `evaluate` checks the seal, rederives fold metrics, selection and thresholds from the OOF predictions, and requires the same recorded dependency/Python versions. An exclusive workspace-level `modelling/test_attempt.json` is created before decoding historical test partitions or scoring any test rows. A failed attempt remains an attempt; do not delete it, switch Gold runs or modify the protocol to obtain another evaluation. Only the selected and prior models are scored once on the identical test population; no-alarm and all-alarm references are computed from those saved scores. There is no test-time fitting, tuning or threshold optimization. The test evidence includes row and event metrics, calendar-event coverage, yearly diagnostics and reliability bins. `verify` checks checksums and reconstructs fold and test metrics from stored prediction rows, without loading an estimator or generating new predictions. Hashes support accidental-change detection, not protection against a user deliberately rewriting both an artifact and its manifest. Use trusted local joblib artifacts only.

## Execution and evidence

Run from the repository root with the pinned requirements and Java 17 or later. Notebook 06 performs the complete offline gate and development; Notebook 07 repeats the gate, verifies the sealed development artifact, and performs the single historical evaluation or read-only verification if already complete. The workspace is `MyDrive/11230945`. No hardcoded upstream run ID is needed. Development evidence lives under `modelling/<Gold manifest SHA-256>/`; the workspace-level attempt marker is shared across Gold runs. Use one writer/runtime at a time. If development fails before sealing and no test attempt exists, preserve the incomplete directory for diagnosis and move it aside only after recording the error; do not overwrite a sealed development result. A test failure after the marker requires documented diagnosis, not silent rescoring. Save executed notebooks with outputs; data, OOF/test predictions and fitted estimators remain in the workspace.

## Sources

- BFT65B final examination instructions and course materials on pipelines, time-ordered validation, purge and imbalanced learning.
- FDIC [Failures and Assistance](https://banks.data.fdic.gov/bankfind-suite/failures/help) and [API field definitions](https://api.fdic.gov/banks/docs/).
- [Scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) and [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
- [LightGBM 4.6.0 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
- [Spark ML pipelines](https://spark.apache.org/docs/4.0.1/ml-pipeline.html).
