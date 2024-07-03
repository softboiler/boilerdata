"""Project paths."""

from pathlib import Path

from boilercore.models import CreatePathsModel
from boilercore.paths import get_package_dir, map_stages
from pydantic.v1 import DirectoryPath, FilePath

import boilerdata
from boilerdata import PROJECT_PATH


class Paths(CreatePathsModel):
    """Paths relevant to the project."""

    # * Roots
    project: DirectoryPath = PROJECT_PATH
    data: DirectoryPath = project / "data"

    # * Local inputs
    propshop: DirectoryPath = data / "propshop"

    # * Git-tracked inputs
    # ! Package
    package: DirectoryPath = get_package_dir(boilerdata)
    axes_enum: FilePath = package / "axes_enum.py"
    models: DirectoryPath = package / "models"
    stages: dict[str, FilePath] = map_stages(
        package / "stages", suffixes=[".py", ".ipynb"]
    )
    validation: FilePath = package / "validation.py"
    # ! Config
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = data / "config"
    axes_config: FilePath = config / "axes.yaml"
    trials_config: FilePath = config / "trials.yaml"
    # ! Plotting
    plot_config: DirectoryPath = config / "plotting"
    mpl_base: FilePath = plot_config / "base.mplstyle"
    mpl_hide_title: FilePath = plot_config / "hide_title.mplstyle"

    # * DVC-tracked imports
    model_functions: Path = data / "models"

    # * DVC-tracked inputs
    benchmarks: DirectoryPath = data / "benchmarks"
    # ! Axes
    axes: DirectoryPath = data / "axes"
    axes_enum_copy: Path = axes / "axes_enum.py"
    file_originlab_coldes: Path = axes / "originlab_coldes.txt"
    # ! Literature
    literature: DirectoryPath = data / "literature"
    # ! Plotter
    plotter: DirectoryPath = data / "plotter"
    file_plotter: FilePath = plotter / "results.opju"
    # ! Trials/curves
    trials: DirectoryPath = data / "curves"

    # * DVC-tracked results
    # ! Benchmarks parsed
    benchmarks_parsed: DirectoryPath = data / "benchmarks_parsed"
    file_benchmarks_parsed: Path = benchmarks_parsed / "benchmarks_parsed.csv"
    # ! Literature results
    literature_results: DirectoryPath = data / "literature_results"
    file_literature_results: Path = literature_results / "lit.csv"
    # ! Originlab plots
    originlab_plots: DirectoryPath = data / "originlab_plots"
    originlab_plot_files: dict[str, Path] = (  # noqa: PLC3002  # need lambda *shrug*
        lambda originlab_plots: {
            shortname: originlab_plots / f"{shortname}.png"
            for shortname in ["lit", "low"]
        }
    )(originlab_plots)
    # ! Originlab results
    originlab_results: DirectoryPath = data / "originlab_results"
    file_originlab_results: Path = originlab_results / "originlab_results.csv"
    # ! Plots
    plots: DirectoryPath = data / "plots"
    plot_new_fit_0: Path = plots / "new_fit_0.png"
    plot_new_fit_1: Path = plots / "new_fit_1.png"
    plot_new_fit_2: Path = plots / "new_fit_2.png"
    plot_error_T_s: Path = plots / "error_T_s.png"  # noqa: N815
    plot_error_q_s: Path = plots / "error_q_s.png"
    plot_error_h_a: Path = plots / "error_h_a.png"
    # ! Results
    results: DirectoryPath = data / "results"
    file_results: Path = results / "results.csv"
    # ! Runs
    runs: DirectoryPath = data / "runs"
    file_runs: Path = runs / "runs.csv"
    # ! Tables
    tables: DirectoryPath = data / "tables"
    file_pipeline_metrics: Path = tables / "pipeline_metrics.json"
