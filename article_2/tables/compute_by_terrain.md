**Table 9.** Computation per synthetic terrain class: full-grid search, CTA search and HAG construction.

| ID | Full grid (ms) | CTA (ms) | Speed-up (×) | HAG build (ms) | Break-even (queries) | Never repaid |
| -------- | ------------ | ------------ | ------------ | ------------ | ------------- | ----------- |
| S1 | 38.3 | 20.2 | 2.5 | 36 | 4 | 0/48 |
| S2 | 33.7 | 11.6 | 4.6 | 83 | 6 | 0/48 |
| S3 | 37.9 | 9.6 | 8.1 | 184 | 15 | 1/48 |
| S4 | 38.3 | 13.7 | 3.4 | 50 | 4 | 0/48 |
| S5 | 41.7 | 9.2 | 6.0 | 126 | 9 | 0/48 |
| S6 | 38.7 | 7.2 | 14.5 | 278 | 16 | 3/48 |
| S7 | 40.1 | 11.7 | 4.3 | 61 | 5 | 0/48 |
| S8 | 39.5 | 7.3 | 9.0 | 158 | 10 | 0/48 |
| S9 | 39.0 | 6.0 | 19.4 | 364 | 15 | 6/48 |
| S10 | 42.7 | 9.7 | 7.2 | 87 | 5 | 0/48 |
| S11 | 40.5 | 4.3 | 17.7 | 228 | 12 | 2/48 |
| S12 | 43.4 | 3.7 | 33.6 | 495 | 22 | 9/48 |
| S13 | 45.3 | 8.6 | 8.5 | 113 | 6 | 0/48 |
| S14 | 42.7 | 2.7 | 23.3 | 295 | 13 | 3/48 |
| S15 | 44.1 | 1.4 | 43.8 | 611 | 27 | 14/48 |
| H1 | 9.8 | 0.7 | 22.8 | 177 | 38 | 16/63 |
| H2 | 11.1 | 0.5 | 28.5 | 174 | 30 | 12/63 |
| H3 | 12.5 | 0.5 | 31.9 | 211 | 29 | 12/54 |

*Times are means per query over a stratified sample of the scenarios of each class (grids of 120, 240, 480 and 960 cells), re-timed one at a time on an otherwise idle machine. Speed-up: full-grid search time divided by CTA search time. Break-even: median number of queries on one terrain after which the time saved repays the HAG construction. Never repaid: queries whose CTA search was not faster than the full-grid search, out of the sample.*
