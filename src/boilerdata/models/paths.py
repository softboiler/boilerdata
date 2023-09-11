"""Project paths."""

from pathlib import Path

from boilercore.models import CreatePathsModel
from boilercore.paths import get_package_dir, map_stages
from pydantic import DirectoryPath, FilePath

import boilerdata
from boilerdata import PROJECT_PATH


class Paths(CreatePathsModel):
    """Paths relevant to the project."""

    # * Roots
    # ! Project
    project: DirectoryPath = PROJECT_PATH
    # ! Package
    package: DirectoryPath = get_package_dir(boilerdata)
    axes_enum: FilePath = package / "axes_enum.py"
    models: DirectoryPath = package / "models"
    stages: dict[str, FilePath] = map_stages(package / "stages", package)
    validation: FilePath = package / "validation.py"
    # ! Data
    data: DirectoryPath = project / "data"
    # ! Config
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = data / "config"

    # * Git-tracked Inputs
    # ! Axes And Trial Configs
    axes_config: FilePath = config / "axes.yaml"
    trials_config: FilePath = config / "trials.yaml"
    # ! Plot Configs
    plot_config: DirectoryPath = config / "plotting"
    # ? Files
    mpl_base: FilePath = plot_config / "base.mplstyle"
    mpl_hide_title: FilePath = plot_config / "hide_title.mplstyle"

    # * Git-tracked results
    test_file_model: Path = project / "tests/root/data/modelfun/model.dillpickle"

    # * Local Inputs
    # ! Properties
    propshop: DirectoryPath = data / "propshop"

    # * DVC-Tracked Inputs
    # ! Benchmarks
    benchmarks: DirectoryPath = data / "benchmarks"
    # ! Plotter
    plotter: DirectoryPath = data / "plotter"
    file_plotter: FilePath = plotter / "results.opju"
    # ! Axes
    axes: DirectoryPath = data / "axes"
    # ? Files
    axes_enum_copy: Path = axes / "axes_enum.py"
    file_originlab_coldes: Path = axes / "originlab_coldes.txt"
    # ! Curves (Trials)
    trials: DirectoryPath = data / "curves"
    # ! Literature
    literature: DirectoryPath = data / "literature"

    # * DVC-Tracked Results
    # ! Benchmarks Parsed
    benchmarks_parsed: DirectoryPath = data / "benchmarks_parsed"
    file_benchmarks_parsed: Path = benchmarks_parsed / "benchmarks_parsed.csv"
    # ! Literature Results
    literature_results: DirectoryPath = data / "literature_results"
    file_literature_results: Path = literature_results / "lit.csv"
    # ! Model Fit Function
    modelfun: DirectoryPath = data / "modelfun"
    file_model: Path = modelfun / "model.dillpickle"
    # ! Originlab Plots
    originlab_plots: DirectoryPath = data / "originlab_plots"
    originlab_plot_files: dict[str, Path] = (  # noqa: PLC3002  # need lambda *shrug*
        lambda originlab_plots: {
            shortname: originlab_plots / f"{shortname}.png"
            for shortname in ["lit", "low"]
        }
    )(originlab_plots)
    # ! Originlab Results
    originlab_results: DirectoryPath = data / "originlab_results"
    file_originlab_results: Path = originlab_results / "originlab_results.csv"
    # ! Plots
    plots: DirectoryPath = data / "plots"
    # ? Files
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
