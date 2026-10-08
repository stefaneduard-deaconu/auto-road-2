"""The reference list of the article, in MDPI numbered style.

Every entry records HOW it was verified. `verified` is one of

    'web'      checked against a publisher, archive or index page during the drafting session
               (the URL is in `source`)
    'source'   taken from a document in the repository (the DEM metadata report, the norm
               transcribed in `data/configs/road_classes.py`, the authors' own papers)
    'TODO'     from memory or not yet checked: the builder lists it in TODO.md and refuses to
               call the reference list final while one remains

The two earlier papers' reference lists contain auto-generated artefacts (entries reading
"undefined", duplicated venue names), so nothing is copied from either: every entry here was
written for this paper.

Keys are the ones used in `{{cite:key}}` markers in `article_2/src/*.md`; the builder numbers
them in order of first citation.
"""
from __future__ import annotations

REFERENCES: dict[str, dict] = {
    'paperA': {
        'text': ('Șomîtcă, I.A.; Deaconu, Ș.E. Development of an Open-Source Library for '
                 'Automated Road Design. *J. Appl. Comput. Sci. Math.* **2025**, *19*(38), '
                 '35–42. https://doi.org/10.4316/JACSM.202501006.'),
        'verified': 'source',
        'source': 'DOI 10.4316/JACSM.202501006',
    },
    'dijkstra1959': {
        'text': ('Dijkstra, E.W. A note on two problems in connexion with graphs. *Numer. Math.* '
                 '**1959**, *1*, 269–271. https://doi.org/10.1007/BF01386390.'),
        'verified': 'web',
        'source': 'link.springer.com/article/10.1007/BF01386390 and ir.cwi.nl/pub/9256 (Numer. Math. 1 (1959), pp. 269-271)',
    },
    'perlin1985': {
        'text': ('Perlin, K. An image synthesizer. *ACM SIGGRAPH Comput. Graph.* **1985**, *19*(3), '
                 '287–296. https://doi.org/10.1145/325165.325247.'),
        'verified': 'web',
        'source': 'dl.acm.org/doi/10.1145/325165.325247 (Computer Graphics 19(3):287-296, July 1985)',
    },
    'botea2004': {
        'text': ('Botea, A.; Müller, M.; Schaeffer, J. Near optimal hierarchical path-finding. '
                 '*J. Game Dev.* **2004**, *1*(1), 7–28.'),
        'verified': 'web',
        'source': 'web search result listing authors, journal, volume, issue and pages 7–28',
    },
    'antikainen2013': {
        'text': ('Antikainen, H. Using the Hierarchical Pathfinding A* Algorithm in GIS to Find '
                 'Paths through Rasters with Nonuniform Traversal Cost. *ISPRS Int. J. Geo-Inf.* '
                 '**2013**, *2*(4), 996–1014. https://doi.org/10.3390/ijgi2040996.'),
        'verified': 'web',
        'source': 'Semantic Scholar and MDPI listing (single author, ISPRS Int. J. Geo-Inf. 2013, '
                  '2, 996-1014)',
    },
    'harabor2011': {
        'text': ('Harabor, D.; Grastien, A. Online graph pruning for pathfinding on grid maps. In '
                 '*Proceedings of the 25th AAAI Conference on Artificial Intelligence*; AAAI '
                 'Press: Palo Alto, CA, USA, 2011; pp. 1114–1119. '
                 'https://doi.org/10.1609/aaai.v25i1.7994.'),
        'verified': 'web',
        'source': 'https://mlanthology.org/aaai/2011/harabor2011aaai-online/ (pages 1114-1119, '
                  'DOI 10.1609/AAAI.V25I1.7994)',
    },
    'uras2013': {
        'text': ('Uras, T.; Koenig, S.; Hernández, C. Subgoal graphs for optimal pathfinding in '
                 'eight-neighbor grids. In *Proceedings of the 23rd International Conference on '
                 'Automated Planning and Scheduling (ICAPS 2013)*; AAAI Press: Palo Alto, CA, '
                 'USA, 2013; pp. 224–232.'),
        'verified': 'web',
        'source': 'web search result (ICAPS 2013, pages 224-232)',
    },
    'daniel2010': {
        'text': ('Daniel, K.; Nash, A.; Koenig, S.; Felner, A. Theta*: Any-angle path planning on '
                 'grids. *J. Artif. Intell. Res.* **2010**, *39*, 533–579.'),
        'verified': 'web',
        'source': 'web search result (JAIR volume 39, pages 533-579, 2010); arXiv:1401.3843',
    },
    'geisberger2008': {
        'text': ('Geisberger, R.; Sanders, P.; Schultes, D.; Delling, D. Contraction hierarchies: '
                 'faster and simpler hierarchical routing in road networks. In *Experimental '
                 'Algorithms (WEA 2008)*; Lecture Notes in Computer Science, Vol. 5038; '
                 'Springer: Berlin/Heidelberg, Germany, 2008; pp. 319–333. '
                 'https://doi.org/10.1007/978-3-540-68552-4_24.'),
        'verified': 'web',
        'source': 'Springer chapter listing (WEA 2008, LNCS 5038, pp. 319-333)',
    },
    'delling2011': {
        'text': ('Delling, D.; Goldberg, A.V.; Pajor, T.; Werneck, R.F. Customizable route '
                 'planning. In *Experimental Algorithms (SEA 2011)*; Lecture Notes in Computer '
                 'Science, Vol. 6630; Springer: Berlin/Heidelberg, Germany, 2011; pp. 376–387. '
                 'https://doi.org/10.1007/978-3-642-20662-7_32.'),
        'verified': 'web',
        'source': 'Springer chapter listing (SEA 2011, LNCS 6630, pp. 376-387)',
    },
    'jong2003': {
        'text': ('Jong, J.-C.; Schonfeld, P. An evolutionary model for simultaneously optimizing '
                 '3-dimensional highway alignments. *Transp. Res. Part B Methodol.* **2003**, '
                 '*37*(2), 107–128.'),
        'verified': 'web',
        'source': 'web search result (Transportation Research Part B, vol. 37, no. 2, pp. 107-128)',
    },
    'jong2000': {
        'text': ('Jong, J.-C.; Jha, M.K.; Schonfeld, P. Preliminary highway design with genetic '
                 'algorithms and geographic information systems. *Comput.-Aided Civ. Infrastruct. '
                 'Eng.* **2000**, *15*(4), 261–271. https://doi.org/10.1111/0885-9507.00190.'),
        'verified': 'web',
        'source': 'structurae.de and teacher.tku.edu.tw listings (CACIE 15(4):261-271, July 2000, '
                  'DOI 10.1111/0885-9507.00190), checked 2026-10-08',
    },
    'jha2004': {
        'text': ('Jha, M.K.; Schonfeld, P. A highway alignment optimization model using geographic '
                 'information systems. *Transp. Res. Part A Policy Pract.* **2004**, *38*(6), '
                 '455–481.'),
        'verified': 'web',
        'source': 'https://ideas.repec.org/a/eee/transa/v38y2004i6p455-481.html, checked 2026-10-08',
    },
    'zhao2019': {
        'text': ('Zhao, L.; Liu, Z.; Mbachu, J. Highway Alignment Optimization: An Integrated BIM and '
                 'GIS Approach. *ISPRS Int. J. Geo-Inf.* **2019**, *8*(4), 172. '
                 'https://doi.org/10.3390/ijgi8040172.'),
        'verified': 'web',
        'source': 'https://www.mdpi.com/2220-9964/8/4/172 and ir.bjut.edu.cn/item/5737, checked 2026-10-08',
    },
    'golab2019': {
        'text': ('Gołąb, J.; Pirowski, T. An attempt at the automation of the routing of mountain '
                 'forest roads with the use of GIS spatial analyses. *Geomat. Environ. Eng.* '
                 '**2019**, *13*(4). https://doi.org/10.7494/geom.2019.13.4.17.'),
        'verified': 'web',
        'source': 'https://gaee.agh.edu.pl/gaee/article/view/40 and repo.agh.edu.pl, checked 2026-10-08',
    },
    'enache2013': {
        'text': ('Enache, A.; Kühmaier, M.; Stampfer, K.; Ciobanu, V.D. An integrative decision '
                 'support tool for assessing forest road options in a mountainous region in '
                 'Romania. *Croat. J. For. Eng.* **2013**, *34*(1), 43–59.'),
        'verified': 'web',
        'source': 'https://hrcak.srce.hr/116728 (Croatian Journal of Forest Engineering 34(1), 2013); '
                  'one index gives pages 43-60, confirm at proofreading. Checked 2026-10-08',
    },
    'stoicafuchs2021': {
        'text': ('Stoica-Fuchs, B.; Mihai, B.-A.; Săvulescu, I.; Dobre, R. Cost-suitability land '
                 'modeling for current and proposed transport infrastructure along Timiș-Cerna '
                 'Corridor (Romania). *Transp. Geogr. Pap. Pol. Geogr. Soc.* **2021**, *24*(1), '
                 '44–56. https://doi.org/10.4467/2543859XPKG.21.003.14946.'),
        'verified': 'web',
        'source': 'ejournals.eu (pkgkptg) and ceeol.com/search/article-detail?id=1012573, checked '
                  '2026-10-08; one index spells the second author "Michai", confirm "Mihai" at '
                  'proofreading',
    },
    'mondal2015': {
        'text': ('Mondal, S.; Lucet, Y.; Hare, W. Optimizing horizontal alignment of roads in a '
                 'specified corridor. *Comput. Oper. Res.* **2015**, *64*, 130–138.'),
        'verified': 'web',
        'source': 'web search result (Computers & Operations Research, vol. 64, pp. 130-138); '
                  'arXiv:1507.02714',
    },
    'pushak2016': {
        'text': ('Pushak, Y.; Hare, W.; Lucet, Y. Multiple-path selection for new highway '
                 'alignments using discrete algorithms. *Eur. J. Oper. Res.* **2016**, *248*(2), '
                 '415–427. https://doi.org/10.1016/j.ejor.2015.07.039.'),
        'verified': 'web',
        'source': 'arXiv:1508.03064 and the IDEAS/RePEc listing (EJOR 248(2):415-427)',
    },
    'vanjeenathammal2026': {
        'text': ('Vanjeenathammal, P.M.; Thompson, J.R.J.; Hare, W.; Lucet, Y. From Corridor '
                 'Selection to Earthwork: A Multi-Stage Framework for Automated Road Design via '
                 'Steiner Trees and Convex Optimization. arXiv **2026**, arXiv:2609.19350 '
                 '(preprint, not peer reviewed).'),
        'verified': 'web',
        'source': 'https://arxiv.org/abs/2609.19350 (submitted 16 September 2026)',
    },
    'hodgson1995': {
        'text': ('Hodgson, M.E. What cell size does the computed slope/aspect angle represent? '
                 '*Photogramm. Eng. Remote Sens.* **1995**, *61*(5), 513–517.'),
        'verified': 'web',
        'source': 'web search result (PE&RS 61(5):513-517, 1995)',
    },
    'zhang1994': {
        'text': ('Zhang, W.; Montgomery, D.R. Digital elevation model grid size, landscape '
                 'representation, and hydrologic simulations. *Water Resour. Res.* **1994**, '
                 '*30*(4), 1019–1028. https://doi.org/10.1029/93WR03553.'),
        'verified': 'web',
        'source': 'Wiley Online Library listing (WRR 30(4):1019-1028, DOI 10.1029/93WR03553)',
    },
    'moulin2024': {
        'text': ('Moulin, A. *Digital Terrain Model of the Idrija Fault in Northwest Slovenia, '
                 '2004*; distributed by OpenTopography, 2024. https://doi.org/10.5069/G9QC01Q2 '
                 '(accessed `TODO(data): access date`).'),
        'verified': 'web',
        'source': 'OpenTopography dataset landing page and citation policy '
                  '(opentopography.org/citations); licence CC0 1.0',
    },
    'moulin2014': {
        'text': ('Moulin, A.; Benedetti, L.; Gosar, A.; Jamšek Rupnik, P.; Rizza, M.; '
                 'Bourlès, D.; Ritz, J.-F. Determining the present-day kinematics of the Idrija '
                 'fault (Slovenia) from airborne LiDAR topography. *Tectonophysics* **2014**, '
                 '*628*, 188–205. https://doi.org/10.1016/j.tecto.2014.04.043.'),
        'verified': 'source',
        'source': 'docs/MetaData_IdrijaLiDAR.pdf, page 1',
    },
    'harris2020': {
        'text': ('Harris, C.R.; Millman, K.J.; van der Walt, S.J.; et al. Array programming with '
                 'NumPy. *Nature* **2020**, *585*, 357–362. '
                 'https://doi.org/10.1038/s41586-020-2649-2.'),
        'verified': 'web',
        'source': 'nature.com/articles/s41586-020-2649-2 (Nature 585:357-362)',
    },
    'virtanen2020': {
        'text': ('Virtanen, P.; Gommers, R.; Oliphant, T.E.; et al. SciPy 1.0: fundamental '
                 'algorithms for scientific computing in Python. *Nat. Methods* **2020**, *17*, '
                 '261–272. https://doi.org/10.1038/s41592-019-0686-2.'),
        'verified': 'web',
        'source': 'nature.com/articles/s41592-019-0686-2 (Nature Methods 17(3):261-272; an Author Correction exists)',
    },
    'ordin1296': {
        'text': ('Ministerul Transporturilor. Ordin nr. 1296/2017 — Norme tehnice din 30 august '
                 '2017 privind proiectarea, construirea și modernizarea drumurilor. *Monitorul '
                 'Oficial al României* **2017**. '
                 'https://legislatie.just.ro/Public/DetaliiDocumentAfis/193251.'),
        'verified': 'source',
        'source': 'data/configs/road_classes.py (the norm and portal URL cited there)',
    },
    'stas863': {
        'text': ('Institutul Român de Standardizare. STAS 863-85 — Lucrări de drumuri. '
                 'Elemente geometrice ale traseelor. Prescripții de proiectare. București, '
                 '1985. `TODO(coauthors): confirm the issuing body, the edition and the table '
                 'numbers of the two design scenarios; the standard is sold by ASRO and the '
                 'code authors did not read an authoritative copy.`'),
        'verified': 'TODO',
        'source': ('the study notes — the two design scenarios '
                   '(V = 40 km/h and V = 25 km/h) as transcribed by the road-design co-author '
                   'on 2026-09-28 and registered as STAS863_V40 / STAS863_V25 in '
                   'data/configs/road_classes.py. A co-author transcription, NOT a reading of '
                   'the standard by the code authors, which is why this entry is TODO.'),
    },
    'bartz2020': {
        'text': ('Bartz-Beielstein, T.; Doerr, C.; van den Berg, D.; et al. Benchmarking in '
                 'optimization: best practice and open issues. arXiv **2020**, arXiv:2007.03488.'),
        'verified': 'web',
        'source': 'https://arxiv.org/abs/2007.03488 (17 authors; eight benchmarking topics)',
    },
    'mdpi_ai': {
        'text': ('MDPI. Instructions for Authors: *Electronics*. '
                 'https://www.mdpi.com/journal/electronics/instructions (accessed '
                 '`TODO(venue): access date`).'),
        'verified': 'web',
        'source': 'MDPI journal instructions and generative-AI authorship policy, as summarised '
                  'by web search: disclosure in Acknowledgments and Materials and Methods; no AI '
                  'authorship; language editing exempt',
    },
}
