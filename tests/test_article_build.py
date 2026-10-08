"""Runs `article_2.build_article`'s checks against the committed results and sources.

`article_2/README.md` promises this file exists ("`tests/test_article_build.py` runs the same
checks on the committed results"). It only reads `article_2/src/*.md` and `results/`, and
writes any output into `tmp_path`, never into `article_2/` itself.
"""
from article_2 import build_article


def _build(tmp_path, **kwargs):
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    sources = sorted((build_article.HERE / 'src').glob('*.md'))
    assert sources, f'no sources in {build_article.HERE / "src"}'
    # figures_dir is the committed `article_2/figures/`, read only (make_figures=False):
    # a figure only becomes a placeholder if it is genuinely missing, not because a fresh
    # tmp_path has nothing in it yet.
    return build_article.build(data, sources, build_article.HERE / 'figures', tmp_path / 'tables',
                               make_figures=False, **kwargs)






def test_an_unknown_fact_name_is_a_build_error(tmp_path):
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    bad_source = tmp_path / '00_bad.md'
    bad_source.write_text('# Bad\n\n{{fact:this_fact_does_not_exist}}\n', encoding='utf-8')
    try:
        build_article.build(data, [bad_source], tmp_path / 'figures', tmp_path / 'tables',
                            make_figures=False)
    except build_article.BuildError as error:
        assert 'this_fact_does_not_exist' in str(error)
    else:
        raise AssertionError('an unknown fact name should have raised BuildError')


def test_every_table_and_figure_has_a_description():
    assert set(build_article.TABLES) <= set(build_article.TABLE_DESCRIPTIONS)
    assert set(build_article.FIGURES) <= set(build_article.FIGURE_DESCRIPTIONS)
    for description, source in [*build_article.TABLE_DESCRIPTIONS.values(),
                                *build_article.FIGURE_DESCRIPTIONS.values()]:
        assert description.strip() and source.strip()


def test_every_terrain_class_has_a_name_and_a_measured_description():
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    for key, tc in build_article.TERRAIN_CLASSES.items():
        assert tc.name.strip(), key
        text = build_article.class_description(data, tc)
        assert 'gon' in text and 'STAS 863-85' in text, key


def test_the_index_lists_every_table_figure_and_class():
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    text = build_article.index_markdown(data)
    for key in [*build_article.TABLES, *build_article.FIGURES, *build_article.TERRAIN_CLASSES]:
        assert f'`{key}`' in text, key
    assert '<<VALUE' not in text


def test_the_reduction_is_reported_per_synthetic_terrain_class(tmp_path):
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    table = build_article.TABLES['x3_by_terrain'](data)
    assert len(table.rows) == 18  # 5 periods x 3 reliefs (full tier) + 3 hilly-tier classes


def test_the_reading_appendix_has_three_tables_and_one_row_per_class():
    data = build_article.Data.load(build_article.DEFAULT_RESULTS)
    text = build_article.appendix_markdown(data)
    assert text.startswith('## Appendix A. Reading the Results')
    for name in ('**Table A1.**', '**Table A2.**', '**Table A3.**'):
        assert name in text, name
    assert '<<VALUE' not in text
    synthetic = [k for k in build_article.TERRAIN_CLASSES if k.startswith(('syn_', 'hil_'))]
    dem = [k for k in build_article.TERRAIN_CLASSES if k.startswith('dem_')]
    c2, c3 = text.split('**Table A2.**')[1].split('**Table A3.**')
    assert sum(1 for line in c2.splitlines() if line.startswith('| ') and '---' not in line) \
        == len(synthetic) + 1  # + header
    assert sum(1 for line in c3.splitlines() if line.startswith('| ') and '---' not in line) \
        == len(dem) + 1


