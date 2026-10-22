# Label-noise evidence pack (auto-generated)

Source: data/audit_triage.csv + results/splits_by_chunk.csv

| set | chunks | concordant | empty | ambiguous | intruder | unusable | unusable % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-train (raw labels) | 2563 | 1837 | 546 | 174 | 6 | 726 | 28.3% |
| C-train (cleaned labels) | 1837 | 1837 | 0 | 0 | 0 | 0 | 0.0% |
| val (raw labels) | 412 | 259 | 125 | 28 | 0 | 153 | 37.1% |
| test (raw labels, used by A and C) | 234 | 172 | 44 | 18 | 0 | 62 | 26.5% |

## Per-species noise of the shared test set

| species | test chunks | concordant | unusable | unusable % |
| --- | --- | --- | --- | --- |
| Arborophila torqueola | 13 | 8 | 5 | 38.5% |
| Cuculus micropterus | 25 | 25 | 0 | 0.0% |
| Cyornis unicolor | 29 | 24 | 5 | 17.2% |
| Glaucidium cuculoides | 9 | 4 | 5 | 55.6% |
| Liocichla phoenicea | 12 | 7 | 5 | 41.7% |
| Pellorneum ruficeps | 70 | 43 | 27 | 38.6% |
| Pnoepyga pusilla | 13 | 13 | 0 | 0.0% |
| Pomatorhinus ruficollis | 24 | 21 | 3 | 12.5% |
| Psilopogon lineatus | 5 | 3 | 2 | 40.0% |
| Psittacula eupatria | 28 | 19 | 9 | 32.1% |
| Rimator malacoptilus | 6 | 5 | 1 | 16.7% |

## Reading of the numbers

- Model A trained on 2563 raw-labelled chunks, of which 726 (28.3%) carry an unusable label (empty / ambiguous / different species).
- Model C trained on 1837 chunks, 100% of them audited as concordant.
- The shared test set holds 234 chunks and 62 (26.5%) of them are themselves mislabelled or unusable, so the reported accuracy is measured against partly wrong ground truth.
- Only 172 test chunks (62) have a verified target bird; that is the subset a clean-test evaluation must use before any claim about absolute accuracy.
