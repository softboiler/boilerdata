<!--
Do *NOT* add changelog entries here!

This changelog is managed by towncrier and is compiled at release time.

See https://github.com/python-attrs/attrs/blob/main/.github/CONTRIBUTING.md#changelog for details.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Calendar Versioning](https://calver.org/). The **first number** of the version is the year. The **second number** is incremented with each release, starting at 1 for each year. The **third number** is for emergencies when we need to start branches for older releases, or for very minor changes.

<!-- towncrier release notes start -->

## [2024.1.1](https://github.com/softboiler/boilerdata/tree/2024.1.1)

### Changes

- Handle model function for all supported Python versions

## Unreleased

- Nothing yet

## 0.0.1

- Implement tests
- Refactor out logic for models containing just file paths and project paths
- Flatten the paths model parameters
- Decouple latest development dependency versions from lower bounds in `pyproject.toml`

## 0.0.0

- Freeze requirements used for pipeline reproduction in `repro.txt` for this release
