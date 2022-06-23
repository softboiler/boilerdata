# boilerdata

Data processing pipeline for a nucleate pool boiling apparatus.

## Overview

The data processing approach taken in this repository started over at [pdpipewrench](https://github.com/blakeNaccarato/pdpipewrench). It was initially conceptualized as a way to outfit [pdpipe](https://github.com/pdpipe/pdpipe) pipelines from configuration files, allowing for Pandas pipeline orchestration with minimal code. I have since adopted a less aggressive tact, where I still separate configuration out into YAML files (constants, file paths, pipeline function arguments, etc.), but pipeline logic is handled in `pipeline.py`. I have also done away with using `pdpipe` in this approach, as it doesn't lend itself particularly well to [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load). Besides, my data processing need is not quite the "flavor" of statistical data science type approaches supported by `pdpipe`.

This new approach maintains the benefits of writing logic in Python, while allowing configuration in files. I am using [Pydantic](https://github.com/samuelcolvin/pydantic) as the interface between my configs and my logic, which allows me to specify allowable values with `Enums` and other typing constructs. Expressing allowable configurations with Pydantic allows for generation of schema for your config files, raising errors on typos or missing keys, for example. I also specify the "shape" of my input and output data in configs, and validate my dataframes with [pandera](https://github.com/pandera-dev/pandera). Once these components are in place, it is easy to implement new functionality in the pipeline.

## Using this approach in your own data process

If you would like to adopt this approach to processing your own data, simply fork/clone this repo and begin swapping my configs and logic for your own, or use a similar architecture for your data processing. The general architecture is such that configuration files are in `src/boilerdata/config`, which are ingested by Pydantic models over in `src/boilerdata/models`, schema are then written to `src/boilerdata/schema`, and `.vscode/settings.json` contains mappings between these schema and the YAML configs they represent. Even though this is all structured as an installable package, it is not really meant to be used outside of an editable install like `pip install -e .`. This allows me to leverage stronger forms of linting, testing, and type-checking available only to installed packages as opposed to a folder full of scripts. It also facilitates more complex import chains across modules. The core logic of the pipeline is in `src/boilerdata/pipeline.py`, which relies on the other modules in `src/boilerdata`. Finally, `src/boilerdata/validation.py` is where Pandera validation is carried out. It is worth noting that a special module `src/boilerdata/models/axes_enum.py` is generated from a config file and allows for auto-completion of column names throughout this package. It is generated by a utility function in `src/boilerdata/utils.py`. 
