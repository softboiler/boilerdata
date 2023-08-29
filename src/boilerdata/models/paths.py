"""Project paths."""

from pathlib import Path

from boilercore.models import CreatePathsModel
from pydantic import DirectoryPath, FilePath

import boilerdata
from boilerdata.models import CWD


class ProjectPaths(CreatePathsModel):
    """Directories relevant to the project."""

    # ! PROJECT
    project: DirectoryPath = CWD

    # ! PACKAGE
    package: DirectoryPath = Path(next(iter(boilerdata.__path__)))
    stages: DirectoryPath = package / "stages"
    models: DirectoryPath = package / "models"
    validation: FilePath = package / "validation.py"

    # ! DATA
    data: DirectoryPath = project / "data"

    # ! PROPERTIES
    propshop: DirectoryPath = data / "propshop"

    # ! CONFIG
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = data / "config"
    axes_config: FilePath = config / "axes.yaml"
    trials_config: FilePath = config / "trials.yaml"

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


class Paths(CreatePathsModel):
    """Directories relevant to the project."""

    # ! PROJECT
    project: DirectoryPath = CWD

    # ! PACKAGE
    package: DirectoryPath = Path(next(iter(boilerdata.__path__)))

    # ! DATA
    data: DirectoryPath = project / "data"

    # ! AXES
    axes: DirectoryPath = data / "axes"
    axes_enum: Path = package / "axes_enum.py"
    axes_enum_copy: Path = axes / "axes_enum.py"
    file_originlab_coldes: Path = axes / "originlab_coldes.txt"

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

    # ! PLOTS
    plots: DirectoryPath = data / "plots"
    plot_new_fit_0: Path = plots / "new_fit_0.png"
    plot_new_fit_1: Path = plots / "new_fit_1.png"
    plot_new_fit_2: Path = plots / "new_fit_2.png"
    plot_error_T_s: Path = plots / "error_T_s.png"  # noqa: N815
    plot_error_q_s: Path = plots / "error_q_s.png"
    plot_error_h_a: Path = plots / "error_h_a.png"

    # ! ORIGINLAB PLOTS
    originlab_plots: DirectoryPath = data / "originlab_plots"
    originlab_plot_files: dict[str, Path] = (  # noqa: PLC3002  # need lambda *shrug*
        lambda originlab_plots: {
            shortname: originlab_plots / f"{shortname}.png"
            for shortname in ["lit", "low"]
        }
    )(originlab_plots)

    # ! TABLES
    tables: DirectoryPath = data / "tables"
    file_pipeline_metrics: Path = tables / "pipeline_metrics.json"
