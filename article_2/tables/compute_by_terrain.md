**Table 9.** Computation per synthetic terrain class: full-grid search, CTA search and HAG construction.

| ID | Full grid (ms) | CTA (ms) | Speed-up (×) | HAG build (ms) | Break-even (queries) | Never repaid |
| -------- | ------------ | ------------ | ------------ | ------------ | ------------- | ----------- |
| S1 | 68.1 | 34.8 | 2.3 | 93 | 4 | 1/57 |
| S2 | 58.3 | 22.5 | 3.8 | 178 | 6 | 0/60 |
| S3 | 66.4 | 17.5 | 6.1 | 390 | 17 | 0/57 |
| S4 | 67.5 | 25.0 | 3.0 | 105 | 4 | 0/60 |
| S5 | 71.2 | 16.1 | 5.1 | 273 | 10 | 0/60 |
| S6 | 63.6 | 12.5 | 9.3 | 594 | 17 | 3/60 |
| S7 | 70.1 | 21.6 | 3.7 | 126 | 4 | 0/60 |
| S8 | 65.7 | 13.8 | 6.5 | 350 | 11 | 0/60 |
| S9 | 70.5 | 10.9 | 11.7 | 777 | 17 | 5/60 |
| S10 | 81.3 | 18.2 | 5.7 | 191 | 5 | 0/60 |
| S11 | 73.0 | 9.2 | 11.8 | 511 | 12 | 1/60 |
| S12 | 70.4 | 7.1 | 17.1 | 1074 | 28 | 9/60 |
| S13 | 79.3 | 16.2 | 6.3 | 247 | 6 | 0/60 |
| S14 | 73.8 | 6.2 | 14.3 | 634 | 15 | 3/60 |
| S15 | 76.7 | 3.6 | 23.4 | 1316 | 32 | 12/60 |
| H1 | 17.4 | 1.7 | 14.4 | 354 | 48 | 16/63 |
| H2 | 20.0 | 1.4 | 17.0 | 362 | 36 | 13/63 |
| H3 | 21.9 | 1.5 | 18.0 | 459 | 32 | 11/54 |

*Times are means per query over a stratified sample of the scenarios of each class (grids of 120, 240, 480 and 960 cells), re-timed one at a time on an otherwise idle machine. Speed-up: full-grid search time divided by CTA search time. Break-even: median number of queries on one terrain after which the time saved repays the HAG construction. Never repaid: queries whose CTA search was not faster than the full-grid search, out of the sample.*
