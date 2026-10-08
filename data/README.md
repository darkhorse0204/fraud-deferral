# Data sources, licences, and known issues

Raw and processed data are **not committed**. `MANIFEST.json` (SHA-256 per file) is committed; `python -m src.data.download elliptic baf` fetches and verifies the Kaggle datasets, and a checksum mismatch aborts the run. BitcoinHeist is downloaded from UCI (see below).

## Elliptic Bitcoin Data Set (Study 1)
- Source: Kaggle `ellipticco/elliptic-data-set` (Elliptic). Weber et al., KDD'19 Workshop on Anomaly Detection in Finance, arXiv:1908.02591.
- Licence: **CC BY-NC-ND 4.0** (Kaggle record; reported by the download). Non-commercial; no derivatives, so neither the parquet cache nor the event-level score tables are redistributed.
- 203,769 transaction nodes, 234,355 edges, 49 time steps (~2 weeks each). 4,545 illicit (class `1`), 42,019 licit (class `2`), 157,205 unknown.
- Features: 166 columns; **the first is the time step itself** and is excluded from all feature sets. `F_local` = 93 (features 2–94), `F_all` = 165 (adds 72 provider-aggregated one-hop features).
- Edges never cross time steps (verified: 0 of 234,355).
- Provider features arrive pre-standardised; how the provider normalised them cannot be audited.
- Labelled nodes are not a random sample: a classifier separates labelled from unlabelled nodes with AUC 0.985 (dev era). Results describe the labelled 23% only.
- No monetary amounts; Study 1 costs are in normalised units.

## Bank Account Fraud (BAF) suite (Study 1 replication)
- Source: Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022`. Jesus et al., NeurIPS 2022 Datasets & Benchmarks, arXiv:2211.13358.
- Licence: **CC BY-NC-SA 4.0** for the data (Kaggle record; reported by the download). The authors' GitHub repository is Apache 2.0, which covers their code, not the data. (An earlier note here listed "CC BY-NC-ND" from a search summary; no primary source states that.)
- **Synthetic**: generated with CTGAN plus differential-privacy noise from a real anonymised dataset. Claims must say "synthetic benchmark", not "real bank data".
- 6 files × 1,000,000 rows, 30 features + `month` (0–7) + `fraud_bool` (≈1.1%).
- `device_fraud_count` is constant: dropped. Five columns use an exact `-1` as the missing sentinel (`prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`); `intended_balcon_amount`, `credit_risk_score`, `velocity_6h` have legitimate negative values. Five categorical columns.
- Protected attributes (age, employment status, income) are present; fairness is out of scope.
- Fraud prevalence drifts in Base (0.92% in month 3 → 1.48% in month 7).

## BitcoinHeist ransomware address dataset (Study 2)
- Source: UCI Machine Learning Repository, https://doi.org/10.24432/C5BG8V (download: `https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip`, extracted to `data/raw/bitcoinheist/`). Akcora et al., IJCAI 2020, pp. 4439–4445.
- Licence: **CC BY 4.0**.
- 2,916,697 rows, one per (address, day), 2009–2018 (rows start in 2011). 41,413 ransomware rows (28 family labels), the rest `white`. `income` is the amount received in satoshis.
- **Legitimate addresses are subsampled to at most 1,000 per day**; ransomware rows are complete. Prevalence is therefore not a population rate, a day's row count leaks the number of positives, and address-repeat counts are a sampling artefact. None of these is used as a feature.
- `white` labels are not verified negatives (dataset documentation).
- The six published features are near chance forward in time (dev AUPRC 0.038, base rate 0.029). The Study 2 scorer adds one matured-blacklist flag (dev AUPRC 0.351); see `src/data/bitcoinheist.py`.
- 2018 has 3 positives and is excluded from the test era.

## Not used
- Kaggle ULB credit-card (no entity ids, 2 days) and PaySim (simulator rules).
- IEEE-CIS: competition data need a Kaggle competition token and acceptance of the competition rules, whose terms for research use were not verified.
