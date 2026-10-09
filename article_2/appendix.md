## Appendix A. Reading the Results

Table A1 defines the terms used in Section 3. Tables A2 and A3 give one mean value per terrain class for each main measure; the classes are defined in Table 4 and the spread of each value is given in Section 3.

**Table A1.** Terms and measures used in the Results.

| Term | Meaning |
| ----------- | -------------------------------- |
| Height area | A connected set of grid cells whose heights fall in the same height interval. |
| Height Area Graph (HAG) | The graph whose nodes are the height areas and whose edges join areas that share a border. |
| Shared-border edge cost | The weight of a HAG edge: the mean height step across the common border of the two areas, priced with the same cost model as the grid search, over the path centroid → border midpoint → centroid. |
| Candidate terrain areas (CTA) | The height areas on the minimum-weight HAG path between the areas of the origin and of the destination; the detailed search runs on their cells only. |
| CTA and one ring | The candidate terrain areas together with every area that borders them. |
| Search space (% of the full grid) | Cells of the restricted search as a share of the cells of the full grid (of the surveyed cells, on the real DEM). |
| Objective ratio | Objective (sum of absolute height differences along the alignment) of the restricted search divided by that of the full-grid search on the same scenario; 1 means no loss. |
| Hausdorff distance | The largest distance from a point of one alignment to the other alignment; it measures how far the restricted alignment departs from the full-grid one. |
| Nodes-expanded ratio | Nodes expanded by the restricted search divided by those of the full-grid search; independent of the machine. |
| Break-even | The number of queries on one terrain after which the time saved per query repays the construction of the HAG; never reached when the restricted search is not faster. |
| Station | A point of the alignment at which every STAS 863-85 parameter is evaluated. |
| Not applicable | A parameter that the road class does not define, or that governs no station of the alignment; it is never counted as a pass. |
| Median slope (gon) | The median terrain slope of a class, in gon (400 gon = 360°); the STAS 863-85 hilly class is 20–25 gon. |

**Table A2.** Mean results per synthetic terrain class.

| ID | CTA (%) | Objective ratio | Nodes ratio | Speed-up (×) | CTA + 1 ring (%) | Objective ratio, + 1 ring |
| -------- | ----------- | ------------ | ----------- | ------------ | ------------ | ------------- |
| S1 | 46.7 | 1.04 | 0.55 | 2.5 | 67.9 | 1.00 |
| S2 | 29.4 | 1.18 | 0.36 | 4.6 | 42.1 | 1.03 |
| S3 | 19.3 | 1.33 | 0.25 | 8.1 | 28.0 | 1.08 |
| S4 | 36.7 | 1.13 | 0.41 | 3.4 | 60.4 | 1.02 |
| S5 | 21.3 | 1.33 | 0.24 | 6.0 | 33.9 | 1.09 |
| S6 | 12.1 | 1.47 | 0.14 | 14.5 | 19.7 | 1.14 |
| S7 | 32.3 | 1.22 | 0.35 | 4.3 | 56.6 | 1.05 |
| S8 | 16.9 | 1.49 | 0.19 | 9.0 | 29.4 | 1.14 |
| S9 | 8.3 | 1.61 | 0.10 | 19.4 | 15.0 | 1.21 |
| S10 | 23.6 | 1.37 | 0.24 | 7.2 | 48.9 | 1.07 |
| S11 | 9.9 | 1.60 | 0.10 | 17.7 | 21.1 | 1.19 |
| S12 | 4.2 | 1.54 | 0.05 | 33.6 | 9.2 | 1.21 |
| S13 | 20.7 | 1.46 | 0.20 | 8.5 | 47.0 | 1.08 |
| S14 | 7.1 | 1.67 | 0.07 | 23.3 | 17.5 | 1.19 |
| S15 | 2.6 | 1.51 | 0.03 | 43.8 | 6.7 | 1.18 |
| H1 | 5.4 | 1.50 | 0.06 | 22.8 | 12.0 | 1.15 |
| H2 | 4.1 | 1.47 | 0.04 | 28.5 | 10.9 | 1.14 |
| H3 | 3.6 | 1.56 | 0.04 | 31.9 | 10.7 | 1.14 |

*CTA (%): share of the grid searched. Speed-up: full-grid search time divided by CTA search time, from the serial timing sample.*

**Table A3.** Mean results per real-DEM class.

| ID | CTA (%) | Objective ratio | Nodes ratio | CTA + 1 ring (%) | Objective ratio, + 1 ring |
| -------- | ----------- | ------------ | ----------- | ------------ | ------------- |
| D1 | 0.69 | 1.10 | 0.026 | 1.97 | 1.02 |
| D2 | 0.68 | 1.24 | 0.025 | 1.79 | 1.05 |
| D3 | 1.04 | 1.82 | 0.041 | 2.48 | 1.26 |
| D4 | 1.54 | 2.72 | 0.058 | 3.54 | 1.54 |
| D5 | 3.01 | 1.08 | 0.047 | 10.29 | 1.02 |
| D6 | 3.06 | 1.19 | 0.046 | 9.67 | 1.03 |
| D7 | 3.90 | 1.54 | 0.058 | 11.38 | 1.12 |
| D8 | 6.84 | 2.03 | 0.095 | 15.87 | 1.22 |
| D9 | 14.57 | 1.71 | 0.198 | 20.85 | 1.22 |

*CTA (%): share of the surveyed cells searched.*
