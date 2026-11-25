# Part 1 protocol and downstream contract

## Acceptance conditions

The source is accepted only when the requested date scope covers all 170 calendar quarters, both reference endpoints are unfiltered and complete, API page totals agree with file rows, all required bank–quarter keys are unique within quarter, and schemas, physical file inventory and SHA-256 values agree. The source run is then sealed and published through `active_source.json`. A source may have no missing quarter but still fail the checks; absence of any individual optional value is not grounds for dropping that observation.

Storage evidence is accepted when the five formats and four workload outputs agree in values, keys, null masks and schema before timing, every timed trial passes result checks, and every saved evidence file matches its checksum. Evidence is associated with the exact source manifest. Candidate fragments for a 2020 year filter and a 2020Q1 filter should be 4 and 1, respectively, out of 170 Hive partitions. This counts files surviving the partition predicate; it does not measure physical bytes read. Format preparation and equality checks are outside measured query time. The source layout is not replaced automatically on the basis of a benchmark: explain the measured trade-offs and retain the verified immutable source for the downstream pipeline.

## Downstream use

Use `fdic_ews.acquisition.resolve_pointer(workspace)` and `verify(source)` to locate and validate the source. Financial files listed in `manifest.json` are the authoritative inventory. `pyarrow.parquet.ParquetFile(path).read()` reads the physical columns without implicitly adding Hive keys; a Dataset scan may additionally expose `year` and `quarter`. `FAILDATE`, `ENDEFYMD`, `ESTYMD`, `INSDATE` and `COSTMOSTRECENTASOF` are text source values and must be parsed under a documented date policy at Part 2. Do not infer historical status from current `ACTIVE`.

## Research constraints carried forward

- Part 2: preserve separate event history and retrospective terminal audits; use `RESTYPE` to distinguish failure from assistance, and retain ambiguous event evidence. Basel zeros and YTD/quarterly reporting semantics require field-specific, historical validation. Do not use retrospective audit status to exclude all past observations of a CERT.
- Part 3: benchmark the verified clean panel across pandas, Polars and Spark with equivalent logical operations. Additional scale-up experiments are a separate synthetic benchmark and cannot enter modelling.
- Part 4: label `1` only for a verified bank failure in `(q, q + 4 quarters]`; mergers are not failures. Determine eligibility at the row-specific horizon. An unrelated event after the horizon cannot invalidate an already determined label. Keep feature availability and outcome evidence separate; when historical publication/revision timestamps are unavailable, explicit lag assumptions and sensitivity checks are required. An as-of risk set at a decision date must not be derived from future activity or full-history filters. The intended forecast-origin convention is to use the latest available reporting quarter as of quarter-end q, subject to the formal research protocol and source evidence. No modelling labels after 2025Q2.
- Part 5: separate chronological development and test with at least four purged quarters, verify at least 50 actual failure events in the test dates, and make all model/feature/threshold decisions using development data only. A historical test period already examined in other research is not automatically an untouched holdout. The official test evaluation is separate from acquisition and storage.

No downstream model results are generated in Part 1.
