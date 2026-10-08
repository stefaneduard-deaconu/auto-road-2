"""Romanian road classes as design constraints for the automatic road design pipeline.

Design requirement: the algorithm reads its constraints
from one `RoadClass` object and from nothing else. No radius, gradient or cell size is
written as a literal inside Algorithm 1, the engineering checks or the experiment runner.

SOURCES AND STATUS
------------------
Every numeric field below is copied from a primary source and cites the document, the
table and the column it came from. Nothing here is written from memory.

Primary source used for all norm-derived entries:

  MT-2017  Ordinul ministrului transporturilor nr. 1296/2017, "NORME TEHNICE din 30
           august 2017 privind proiectarea, construirea si modernizarea drumurilor"
           (Monitorul Oficial; the copy read is the one published by CNADNR/CNAIR at
           https://www.cnadnr.ro/sites/default/files/Reglementari-tehnice/ORDIN%201296%20proiectarea,%20construirea%20%C5%9Fi%20modernizarea%20drumurilor.pdf
           and on the legislative portal at
           https://legislatie.just.ro/Public/DetaliiDocumentAfis/193251 ).
           This normative replaced the "Norme tehnice privind proiectarea, construirea
           si modernizarea drumurilor" approved by Ordinul MT nr. 45/1998, which was
           repealed on 18 September 2017.

  Tabelul nr. 1 a)  "Vitezele de proiectare pentru diferite clase tehnice ale drumurilor
                    publice", columns "Viteza de proiectare minima (km/h)" / "ses",
                    "deal", "munte", and "Latimea benzii de circulatie (m)".
  Tabelul nr. 1 b)  "Vitezele de proiectare reduse pentru diferite clase tehnice ale
                    drumurilor publice", same columns.
  Tabelul nr. 2 A)  "Elemente geometrice", "Clasa tehnica I - Autostrazi": rows
                    "Razele minime ale curbelor in plan" (m) and "Declivitati
                    longitudinale maxime" (%), per design speed 140/120/100/80 km/h.
  Tabelul nr. 2 B)  "Elemente geometrice", "Clasele tehnice II-V": rows "Razele minime
                    ale curbelor in plan" (m), "Razele minime in serpentine" (m) and
                    "Declivitati longitudinale maxime" (%), per design speed
                    120/100/80/60/50/40/30/25 km/h.

  Tabelul nr. 2 B), transcribed here for traceability:

      Viteza de proiectare (km/h)        120   100    80    60    50    40    30    25
      Razele minime ale curbelor in plan 650   450   240   125    95    60    35    25
      Razele minime in serpentine          -     -     -    30    25    20    20    20
      Declivitati longitudinale maxime     5     5     6   6,5     7     7   7,5     8

  Tabelul nr. 2 A), transcribed here for traceability:

      Viteza de proiectare (km/h)        140   120   100    80
      Razele minime ale curbelor in plan 1000  650   450   240
      Declivitati longitudinale maxime      3     4     5     6

STAS 863-85 ("Lucrari de drumuri. Elemente geometrice ale traseelor. Prescriptii de
proiectare") is the standard cited by both source articles. Its numeric tables could NOT
be confirmed from an authoritative copy on the open web: the standard is sold by ASRO and
the copies reachable there are user uploads on document-sharing sites, which are not an
acceptable citation. On 2026-09-28 the road-design co-author supplied a transcription of
the two design scenarios the article uses; it is recorded in
the study notes (Image 1) and is the ONLY source of the two
`STAS863_*` entries below. It is a co-author transcription, not a reading of the standard,
so those entries are `TO CONFIRM` like every other one. See the `UNRESOLVED` note at the
bottom of this module.

OFFICIAL VERSUS UNOFFICIAL (2026-09-28)
-------------------------------------------------------
The article's design parameters are the STAS 863-85 ones. Every other entry here is
derived from MT-2017 or invented, and is therefore **unofficial for that article's
purposes** - `RoadClass.official` is False for all of them. They are kept, not deleted, so
the result files already committed under those names stay interpretable.

EVERY ENTRY IS MARKED `TO CONFIRM` and stays that way until the road-design co-author
signs it off. `RoadClass.confirmed` is the flag; `assert_confirmed()` is provided so a
publication-grade run can refuse unconfirmed constraints.

Two synthetic classes (`SYNTHETIC_STRICT`, `SYNTHETIC_LOOSE`) are included so the
robustness matrix has a spread. They are NOT norms and are marked `synthetic=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

TerrainCategory = Literal['ses', 'deal', 'munte', 'n/a']

#: the grid cell size used by the two source articles (100x100 grid over 1x1 km).
DEFAULT_CELL_SIZE_M = 10.0


class UnconfirmedRoadClass(RuntimeError):
    """Raised when a run that requires signed-off constraints is given a `TO CONFIRM` class."""


@dataclass(frozen=True)
class RoadClass:
    """One set of geometric design constraints.

    All lengths are in meters, gradients in percent, speeds in km/h.

    Attributes:
        name: registry key / short label used in result files.
        source: the norm, the table and the column every number came from.
        design_speed_kmh: "viteza de proiectare".
        i_max_percent: "declivitate longitudinala maxima", the largest allowed
            longitudinal gradient of the road axis.
        r_min_m: "raza minima a curbei in plan", the smallest allowed horizontal
            curve radius.
        cell_size_m: terrain grid resolution the constraints are applied at. This
            replaces the `GRID_RATIO_TO_METERS` / `SCALE` / `sq_size` literals.
        terrain_category: 'ses', 'deal' or 'munte'; 'n/a' for the synthetic classes.
        technical_class: 'I'..'V' for the norm-derived classes, None otherwise.
        lane_width_m: "latimea benzii de circulatie", where the norm gives it.
        r_min_serpentine_m: "raza minima in serpentine", where the norm gives it.
        l_alignment_min_m: "lungimea minima a aliniamentului", the shortest tangent
            allowed between two successive curves. None when the source does not give it.
        l_design_step_min_m: "pasul de proiectare minim", the spacing of the design
            points of the longitudinal profile. None when the source does not give it.
        r_vert_concave_min_m: minimum concave (valley) vertical-curve radius.
        r_vert_convex_min_m: minimum convex (crest) vertical-curve radius.
        official: True only for the entries the article designs against
            (STAS 863-85). Every other entry is unofficial for that purpose.
        confirmed: False until the road-design co-author signs the entry off.
        synthetic: True for the deliberately strict/loose classes, which are not norms.
        notes: anything the norm requires that the caller should know about.
    """

    name: str
    source: str
    design_speed_kmh: float
    i_max_percent: float
    r_min_m: float
    cell_size_m: float = DEFAULT_CELL_SIZE_M
    terrain_category: TerrainCategory = 'deal'
    technical_class: Optional[str] = None
    lane_width_m: Optional[float] = None
    r_min_serpentine_m: Optional[float] = None
    l_alignment_min_m: Optional[float] = None
    l_design_step_min_m: Optional[float] = None
    r_vert_concave_min_m: Optional[float] = None
    r_vert_convex_min_m: Optional[float] = None
    official: bool = False
    confirmed: bool = False
    synthetic: bool = False
    notes: str = ''

    @property
    def status(self) -> str:
        return 'CONFIRMED' if self.confirmed else 'TO CONFIRM'

    @property
    def standing(self) -> str:
        """'official' for the STAS 863-85 entries, 'unofficial' for everything else."""
        return 'official' if self.official else 'unofficial'

    @property
    def has_vertical_geometry(self) -> bool:
        """Whether the class defines the design step and both vertical-curve minima."""
        return (self.l_design_step_min_m is not None
                and self.r_vert_concave_min_m is not None
                and self.r_vert_convex_min_m is not None)

    @property
    def r_min_cells(self) -> float:
        """`r_min_m` expressed in grid cells, for code that works in grid units."""
        return self.r_min_m / self.cell_size_m

    def with_cell_size(self, cell_size_m: float) -> 'RoadClass':
        """Same constraints on a terrain with a different resolution (e.g. a 30 m DEM)."""
        return replace_cell_size(self, cell_size_m)

    def assert_confirmed(self) -> None:
        if not self.confirmed:
            raise UnconfirmedRoadClass(
                f"road class {self.name!r} is still TO CONFIRM (source: {self.source})")


def replace_cell_size(road_class: RoadClass, cell_size_m: float) -> RoadClass:
    import dataclasses
    return dataclasses.replace(road_class, cell_size_m=cell_size_m)


# --------------------------------------------------------------------------------------
# Norm-derived entries. Emphasis on hilly terrain ("deal"), which is what the article
# models. Each entry's `source` names the exact table and column.
# --------------------------------------------------------------------------------------

_MT2017 = ('MT-2017 = Ordin MT 1296/2017, Norme tehnice din 30.08.2017 privind '
           'proiectarea, construirea si modernizarea drumurilor')

ROAD_CLASSES: dict[str, RoadClass] = {}


def _register(rc: RoadClass) -> RoadClass:
    ROAD_CLASSES[rc.name] = rc
    return rc


RO_CLASS_II_DEAL = _register(RoadClass(
    name='RO_CLASS_II_DEAL',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica II, coloana "deal" -> 100 km/h; '
            'Tabelul nr. 2 B) coloana 100 km/h -> raza minima in plan 450 m, '
            'declivitate longitudinala maxima 5%; Tabelul nr. 1 a) latimea benzii 3,50 m'),
    design_speed_kmh=100,
    i_max_percent=5.0,
    r_min_m=450.0,
    terrain_category='deal',
    technical_class='II',
    lane_width_m=3.50,
    r_min_serpentine_m=None,  # Tabelul nr. 2 B) gives "-" for 100 km/h
    notes='Drumuri expres si drumuri nationale europene cu 4 benzi (Tabelul nr. 1 a).',
))

RO_CLASS_III_DEAL = _register(RoadClass(
    name='RO_CLASS_III_DEAL',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica III, coloana "deal" -> 80 km/h; '
            'Tabelul nr. 2 B) coloana 80 km/h -> raza minima in plan 240 m, '
            'declivitate longitudinala maxima 6%; Tabelul nr. 1 a) latimea benzii 3,50 m'),
    design_speed_kmh=80,
    i_max_percent=6.0,
    r_min_m=240.0,
    terrain_category='deal',
    technical_class='III',
    lane_width_m=3.50,
    r_min_serpentine_m=None,  # Tabelul nr. 2 B) gives "-" for 80 km/h
    notes=('Drumuri nationale europene, drumuri nationale principale si drumuri judetene '
           '(Tabelul nr. 1 a).'),
))

RO_CLASS_IV_DEAL = _register(RoadClass(
    name='RO_CLASS_IV_DEAL',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica IV, coloana "deal" -> 60 km/h; '
            'Tabelul nr. 2 B) coloana 60 km/h -> raza minima in plan 125 m, raza minima '
            'in serpentine 30 m, declivitate longitudinala maxima 6,5%; '
            'Tabelul nr. 1 a) latimea benzii 3,00 m'),
    design_speed_kmh=60,
    i_max_percent=6.5,
    r_min_m=125.0,
    terrain_category='deal',
    technical_class='IV',
    lane_width_m=3.00,
    r_min_serpentine_m=30.0,
    notes=('Drumuri nationale principale, drumuri nationale secundare, drumuri judetene '
           'si drumuri comunale (Tabelul nr. 1 a).'),
))

RO_CLASS_V_DEAL = _register(RoadClass(
    name='RO_CLASS_V_DEAL',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica V, coloana "deal" -> 50 km/h; '
            'Tabelul nr. 2 B) coloana 50 km/h -> raza minima in plan 95 m, raza minima '
            'in serpentine 25 m, declivitate longitudinala maxima 7%; '
            'Tabelul nr. 1 a) latimea benzii 2,75 m'),
    design_speed_kmh=50,
    i_max_percent=7.0,
    r_min_m=95.0,
    terrain_category='deal',
    technical_class='V',
    lane_width_m=2.75,
    r_min_serpentine_m=25.0,
    notes=('Drumuri nationale secundare, drumuri judetene, drumuri comunale si drumuri '
           'vicinale (Tabelul nr. 1 a).'),
))

RO_CLASS_V_DEAL_REDUS = _register(RoadClass(
    name='RO_CLASS_V_DEAL_REDUS',
    source=(f'{_MT2017}; Tabelul nr. 1 b) "viteze de proiectare reduse", clasa tehnica V, '
            'coloana "deal" -> 40 km/h; Tabelul nr. 2 B) coloana 40 km/h -> raza minima '
            'in plan 60 m, raza minima in serpentine 20 m, declivitate longitudinala '
            'maxima 7%; Tabelul nr. 1 b) latimea benzii 2,75 m'),
    design_speed_kmh=40,
    i_max_percent=7.0,
    r_min_m=60.0,
    terrain_category='deal',
    technical_class='V',
    lane_width_m=2.75,
    r_min_serpentine_m=20.0,
    notes=('Reduced design speed. NOTA 2.4.2 of MT-2017: using the reduced geometric '
           'elements requires the road administrator\'s approval on the basis of a '
           'techno-economic study. The loosest norm-derived hilly class in this registry.'),
))

RO_CLASS_I_DEAL_AUTOSTRADA = _register(RoadClass(
    name='RO_CLASS_I_DEAL_AUTOSTRADA',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica I, coloana "deal" -> 120 km/h; '
            'Tabelul nr. 2 A) coloana 120 km/h -> raza minima in plan 650 m, '
            'declivitate longitudinala maxima 4%; Tabelul nr. 1 a) latimea benzii 3,75 m'),
    design_speed_kmh=120,
    i_max_percent=4.0,
    r_min_m=650.0,
    terrain_category='deal',
    technical_class='I',
    lane_width_m=3.75,
    notes=('Autostrazi. The strictest norm-derived hilly class in this registry; on a '
           '1x1 km Perlin terrain it is expected to be infeasible and must be reported '
           'as such, not relaxed.'),
))

# Two non-hilly entries, kept so the effect of the terrain category is visible.
RO_CLASS_III_SES = _register(RoadClass(
    name='RO_CLASS_III_SES',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica III, coloana "ses" -> 100 km/h; '
            'Tabelul nr. 2 B) coloana 100 km/h -> raza minima in plan 450 m, '
            'declivitate longitudinala maxima 5%'),
    design_speed_kmh=100,
    i_max_percent=5.0,
    r_min_m=450.0,
    terrain_category='ses',
    technical_class='III',
    lane_width_m=3.50,
))

RO_CLASS_V_MUNTE = _register(RoadClass(
    name='RO_CLASS_V_MUNTE',
    source=(f'{_MT2017}; Tabelul nr. 1 a) clasa tehnica V, coloana "munte" -> 40 km/h; '
            'Tabelul nr. 2 B) coloana 40 km/h -> raza minima in plan 60 m, raza minima '
            'in serpentine 20 m, declivitate longitudinala maxima 7%'),
    design_speed_kmh=40,
    i_max_percent=7.0,
    r_min_m=60.0,
    terrain_category='munte',
    technical_class='V',
    lane_width_m=2.75,
    r_min_serpentine_m=20.0,
))

# --------------------------------------------------------------------------------------
# STAS 863-85. The two design scenarios of the article, transcribed by the
# road-design co-author on 2026-09-28 (the study notes,
# Image 1). These are the only entries with `official=True`, and the only ones that carry
# the alignment and vertical-curve minima. Both were signed off by the road-design
# co-author on 2026-10-08, so they are CONFIRMED.
# --------------------------------------------------------------------------------------

_STAS863 = ('STAS 863-85 "Lucrari de drumuri. Elemente geometrice ale traseelor. '
            'Prescriptii de proiectare"; values as transcribed by the road-design '
            'co-author on 2026-09-28, recorded in '
            'the study notes (Image 1). Not read from an '
            'authoritative copy of the standard by the code authors.')

STAS863_V40 = _register(RoadClass(
    name='STAS863_V40',
    source=f'{_STAS863} Scenario "baseline", design speed 40 km/h.',
    design_speed_kmh=40,
    i_max_percent=7.0,
    r_min_m=60.0,
    terrain_category='deal',
    l_alignment_min_m=56.0,
    l_design_step_min_m=50.0,
    r_vert_concave_min_m=1000.0,
    r_vert_convex_min_m=1000.0,
    official=True,
    confirmed=True,
    notes=('Baseline scenario of the article. The design step is the distance '
           'between two successive vertical connections, which is what the preliminary '
           '(first-step) design works at.'),
))

STAS863_V25 = _register(RoadClass(
    name='STAS863_V25',
    source=f'{_STAS863} Scenario "difficult conditions", design speed 25 km/h.',
    design_speed_kmh=25,
    i_max_percent=8.0,
    r_min_m=25.0,
    terrain_category='munte',
    l_alignment_min_m=35.0,
    l_design_step_min_m=50.0,
    r_vert_concave_min_m=300.0,
    r_vert_convex_min_m=500.0,
    official=True,
    confirmed=True,
    notes=('Difficult-condition scenario of the article: mountainous or otherwise '
           'constrained terrain, reduced design speed.'),
))


# --------------------------------------------------------------------------------------
# Synthetic classes. NOT norms. They exist only to give the robustness matrix a spread
# beyond what the Romanian tables cover, and they must never be presented as standards.
# --------------------------------------------------------------------------------------

SYNTHETIC_STRICT = _register(RoadClass(
    name='SYNTHETIC_STRICT',
    source='SYNTHETIC - not a norm. Chosen by the authors to bracket the norm values '
           'from above (stricter than every hilly entry except the autostrada).',
    design_speed_kmh=90,
    i_max_percent=3.0,
    r_min_m=300.0,
    terrain_category='n/a',
    synthetic=True,
    notes='Deliberately strict. Used to show the pipeline reports infeasibility instead '
          'of relaxing the constraint.',
))

SYNTHETIC_LOOSE = _register(RoadClass(
    name='SYNTHETIC_LOOSE',
    source='SYNTHETIC - not a norm. Chosen by the authors to bracket the norm values '
           'from below.',
    design_speed_kmh=25,
    i_max_percent=10.0,
    r_min_m=25.0,
    terrain_category='n/a',
    synthetic=True,
    notes=('Deliberately loose. The 10% gradient matches the claim in both source '
           'articles; note that no hilly class in MT-2017 Tabelul nr. 2 allows 10%, the '
           'loosest value in that table being 8% at 25 km/h.'),
))


# --------------------------------------------------------------------------------------
# Proposal for the T7 robustness matrix (3 constraint sets). NOT a decision: the
# road-design co-author picks. Rationale is in the study protocol.
# --------------------------------------------------------------------------------------

PROPOSED_MATRIX_CLASSES: tuple[str, str, str] = (
    'RO_CLASS_III_DEAL',   # strict but plausible hilly class: 80 km/h, R>=240 m, i<=6%
    'RO_CLASS_IV_DEAL',    # the realistic middle: 60 km/h, R>=125 m, i<=6.5%
    'RO_CLASS_V_DEAL_REDUS',  # the loosest norm-derived hilly class: 40 km/h, R>=60 m, i<=7%
)

#: Synthetic classes to add if the co-author wants a wider spread than the norms give.
PROPOSED_MATRIX_SYNTHETIC: tuple[str, str] = ('SYNTHETIC_STRICT', 'SYNTHETIC_LOOSE')

#: The two design scenarios the article evaluates against (STAS 863-85).
STAS_MATRIX_CLASSES: tuple[str, str] = ('STAS863_V40', 'STAS863_V25')


def get(name: str) -> RoadClass:
    try:
        return ROAD_CLASSES[name]
    except KeyError:
        raise KeyError(f"unknown road class {name!r}; known: {sorted(ROAD_CLASSES)}") from None


def unconfirmed() -> list[str]:
    """Names of every entry still waiting for the road-design co-author's sign-off."""
    return sorted(n for n, rc in ROAD_CLASSES.items() if not rc.confirmed)


def unofficial() -> list[str]:
    """Names of the entries that are not the article's STAS 863-85 design set."""
    return sorted(n for n, rc in ROAD_CLASSES.items() if not rc.official)


UNRESOLVED = """\
Gaps that T0 could not close and that the road-design co-author must settle:

1. STAS 863-85. Partly closed on 2026-09-28: the co-author transcribed the two design
   scenarios the article uses, and they are the STAS863_V40 / STAS863_V25 entries
   above. Still open: the transcription was not checked against an authoritative copy of
   the standard by the code authors, and the articles' "slopes under 10%" claim is still
   unsupported - the transcribed scenarios top out at 8%, and MT-2017 Tabelul nr. 2 B)
   also tops out at 8%, at 25 km/h. Open questions: (a) does STAS 863-85 give a hilly
   gradient limit that reaches 10%, and (b) is 863/1-85 a separate part with its own
   tables?
2. Exceptional gradients. MT-2017 Tabelul nr. 2 lists a single "declivitate longitudinala
   maxima" per design speed. Whether an exceptional increment is allowed on short
   sections (older Romanian practice allowed one) is not settled by the tables read here.
3. Vertical geometry. Modelled since 2026-09-28 for the two STAS863_* entries only
   (`l_design_step_min_m`, `r_vert_concave_min_m`, `r_vert_convex_min_m`, plus
   `l_alignment_min_m` in plan). The MT-2017 entries leave them None, which makes the
   corresponding checks report 'not applicable' rather than pass. MT-2017 Tabelul nr. 2
   also fixes minimum sight distances, which are still not modelled.
4. Whether the "viteze de proiectare reduse" (Tabelul nr. 1 b) entries may be used at all
   in the evaluation: NOTA 2.4.2 makes them conditional on the road administrator's
   approval and a techno-economic study.
"""
