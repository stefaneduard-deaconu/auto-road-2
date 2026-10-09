**Table 11.** STAS 863-85 compliance by the edge cost of the detailed search.

| Edge cost of the search | Alignments | Gradient passes | All parameters pass |
| ---------------------------------- | ------------ | ------------ | ------------- |
| Height change | 468 | 45.9% | 8.8% |
| Height change + length tie-break | 468 | 46.8% | 7.9% |
| Length along the terrain | 468 | 41.7% | 20.7% |
| Height change + length | 468 | 41.0% | 23.5% |
| Height change + gradient penalty | 468 | 48.9% | 8.1% |
| Height change, edges steeper than i_max removed | 338 | 68.3% | 11.2% |

*Alignments that found a path, before and after Algorithm 1, both design speeds. The gradient penalty adds one metre of cost per metre of edge for every percent above i_max; removing steep edges can disconnect the selected areas, which is why that row has fewer alignments.*
