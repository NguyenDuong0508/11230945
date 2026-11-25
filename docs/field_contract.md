# Source field selection

The field lists in `src/fdic_ews/contracts.py` are the executable contract. The FDIC BankFind dictionaries are the authoritative definitions. The groups below explain why the selected variables are retained; they are not assertions that a variable has unchanged historical reporting semantics or is usable at a given forecast date.

## Financials (34 fields)

| Fields | Economic role and downstream use |
|---|---|
| `CERT`, `REPDTE`, `STNAME` | Bank–quarter key, period alignment, geographic grouping. State is retained as reported in Financials. |
| `ASSET`, `DEP`, `DEPDOM`, `EQ` | Size, funding, equity buffer, accounting checks and denominators. |
| `BRO`, `CHBAL`, `SC` | Brokered funding, cash/balances and securities exposures; liquidity and funding proxies. |
| `LNLSGR`, `LNLSNET`, `LNRE`, `LNCI`, `LNCON`, `LNATRES` | Gross/net loans, real-estate, commercial and consumer exposure, loan-loss allowance; credit concentration and reserve coverage. |
| `NCLNLS`, `P3ASSET`, `P9ASSET`, `NAASSET`, `NTLNLS`, `ORE` | Noncurrent, past-due, nonaccrual and other real estate indicators. Verify overlaps before forming a ratio; do not add overlapping delinquency categories. |
| `INTAN` | Intangible assets; candidate adjustment to tangible equity for a Texas-like risk ratio. Confirm historical availability and accounting definition before use. |
| `NETINC`, `INTINC`, `EINTEXP`, `NONII`, `NONIX`, `ELNATR` | Income, expense and credit-loss-related flows; possible profitability and cost efficiency features. Establish quarterly vs year-to-date semantics for each field and period before differencing. |
| `ROA`, `ROE`, `NIMY` | Published profitability ratios. Check annualization and denominator conventions; do not treat an extreme ROE as automatically erroneous. |
| `RBC1AAJ`, `RBCRWAJ` | Published risk-based capital ratios. Zero, null and non-applicable values must be assessed by reporting regime before transformations. |

Amounts are generally in thousands of USD, ratios in percentage units, as specified by the exam; field-level metadata takes precedence. `CERT` is int64, `REPDTE` is date32, `STNAME` is string, and remaining Financials fields are float64. `CERT` and `REPDTE` are required. Float64 is appropriate for amounts in the dataset's stated scale and ratio calculations; raw numeric blanks become null and invalid/nonfinite numeric values stop acquisition. No numeric value is divided by 100 or transformed into an economic feature at this stage.

## Failures (12 fields)

`CERT`, `FIN`, `NAME`, `CITYST`, `FAILDATE`, `FAILYR`, `COST`, `QBFASSET`, `QBFDEP`, `RESTYPE`, `RESTYPE1`, `COSTMOSTRECENTASOF`.

The complete endpoint population is acquired, not only the subset with `RESTYPE=FAILURE`. `FIN` is the FDIC failure-event identifier; `RESTYPE` and `RESTYPE1` are retained for event taxonomy and resolution-type audit. `COST` and cost vintage are available for later development-only cost analysis; the cost of a resolution is not a measured cost of missing a quarterly warning. Missing CERTs and repeated CERTs are preserved. Source dates in reference records remain text so questionable values can be adjudicated later without silently converting them into missing dates.

## Institutions (9 fields)

`CERT`, `NAME`, `STNAME`, `BKCLASS`, `ACTIVE`, `ESTYMD`, `ENDEFYMD`, `CHANGEC1`, `INSDATE`.

Used for identity, coverage and dated institutional exit audit. These data describe the acquired snapshot; the present-day `ACTIVE`, `BKCLASS` or `CHANGEC1` fields are not historical predictors. Historical charter class requires independent point-in-time evidence.

## Validation and limitations

Every requested field has absent-key and normalized-null counts in the manifest. A field absent from every record is a source-contract failure, not evidence that its economic value is zero. Explicit nulls remain allowed in non-key fields; historical coverage and reporting conventions are investigated in Part 2. The official dictionaries must be consulted before constructing features from candidate fields. Current API data may contain later revisions to historical reports, and Part 1 does not establish actual publication timestamps or historical vintages.

Sources: [FDIC API](https://api.fdic.gov/banks/docs/), [Financials dictionary](https://api.fdic.gov/banks/docs/risview_properties.yaml), [Failures dictionary](https://api.fdic.gov/banks/docs/failure_properties.yaml), [Institutions dictionary](https://api.fdic.gov/banks/docs/institution_properties.yaml), BFT65B course starter script and examination instructions.
