from pathlib import Path

from pydantic import DirectoryPath, FilePath

from boilerdata import DATA_DIR, PARAMS_FILE, PROJECT_CONFIG, PROJECT_DIR
from boilerdata.models import CreatePathsModel


class ProjectPaths(CreatePathsModel):
    """Directories relevant to the project."""

    # ! REQUIREMENTS
    dev_requirements: DirectoryPath = PROJECT_DIR / ".tools/requirements"

    # ! CONFIG
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = PROJECT_CONFIG

    # ! PACKAGE
    package: DirectoryPath = PROJECT_DIR / "src/boilerdata"
    stages: DirectoryPath = package / "stages"
    models: DirectoryPath = package / "models"
    validation: FilePath = package / "validation.py"

    # ! PLOT CONFIG
    plot_config: DirectoryPath = config / "plotting"
    mpl_base: FilePath = plot_config / "base.mplstyle"
    mpl_hide_title: FilePath = plot_config / "hide_title.mplstyle"

    # ! STAGES
    stage_axes: FilePath = stages / "axes.py"
    stage_parse_benchmarks: FilePath = stages / "parse_benchmarks.py"
    stage_literature: FilePath = stages / "literature.py"
    stage_metrics: FilePath = stages / "metrics.ipynb"
    stage_modelfun: FilePath = stages / "modelfun.ipynb"
    stage_originlab: FilePath = stages / "originlab.py"
    stage_pipeline: FilePath = stages / "pipeline.py"
    stage_runs: FilePath = stages / "runs.py"
    stage_schema: FilePath = stages / "schema.py"


class Paths(CreatePathsModel):
    """Directories relevant to the project."""

    # ! PROJECT FILE
    file_proj: FilePath = PARAMS_FILE

    # ! DATA
    data: DirectoryPath = DATA_DIR

    # ! AXES
    axes: DirectoryPath = data / "axes"
    file_originlab_coldes: Path = axes / "originlab_coldes.txt"

    # ! SCHEMA
    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = data / "schema"

    # ! LITERATURE
    literature: DirectoryPath = data / "literature"

    # ! LITERATURE RESULTS
    literature_results: DirectoryPath = data / "literature_results"
    file_literature_results: Path = literature_results / "lit.csv"

    # ! MODEL FUNCTION
    modelfun: DirectoryPath = data / "modelfun"
    file_model: Path = modelfun / "model.dillpickle"

    # ! TRIALS
    trials: DirectoryPath = data / "curves"

    # ! BENCHMARKS
    benchmarks: DirectoryPath = data / "benchmarks"

    # ! RUNS
    runs: DirectoryPath = data / "runs"
    file_runs: Path = runs / "runs.csv"

    # ! RUNS WITH BENCHMARKS
    benchmarks_parsed: DirectoryPath = data / "benchmarks_parsed"
    file_benchmarks_parsed: Path = benchmarks_parsed / "benchmarks_parsed.csv"

    # ! RESULTS
    results: DirectoryPath = data / "results"
    file_results: Path = results / "results.csv"

    # ! PLOTTER
    plotter: DirectoryPath = data / "plotter"
    file_plotter: FilePath = plotter / "results.opju"

    # ! ORIGINLAB RESULTS
    originlab_results: DirectoryPath = data / "originlab_results"
    file_originlab_results: Path = originlab_results / "originlab_results.csv"

    # ! TOP-LEVEL METRICS DIR
    metrics: DirectoryPath = data / "metrics"

    # ! PLOTS
    plots: DirectoryPath = metrics / "plots"
    plot_new_fit_0: Path = plots / "new_fit_0.png"
    plot_new_fit_1: Path = plots / "new_fit_1.png"
    plot_new_fit_2: Path = plots / "new_fit_2.png"
    plot_error_T_s: Path = plots / "error_T_s.png"  # noqa: N815
    plot_error_q_s: Path = plots / "error_q_s.png"
    plot_error_h_a: Path = plots / "error_h_a.png"

    # ! ORIGINLAB PLOTS
    originlab_plots: DirectoryPath = metrics / "originlab_plots"
    originlab_plot_files: dict[str, Path] = (  # noqa: PLC3002  # need lambda *shrug*
        lambda originlab_plots: {
            shortname: originlab_plots / f"{shortname}.png"
            for shortname in ["lit", "low"]
        }
    )(originlab_plots)

    # ! TABLES
    tables: DirectoryPath = metrics / "tables"
    file_pipeline_metrics: Path = tables / "pipeline_metrics.json"
