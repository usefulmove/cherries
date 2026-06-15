---
title: fanuc_comms ROS 2 Package Setup
description: Python setuptools configuration for the fanuc_comms ROS 2 (ament) package,
  which provides communication utilities for Fanuc robots and registers a console
  script entry point.
source: docs/system_architecture/fanuc_comms/setup.md
tags:
- ros2
- ament
- robotics
- fanuc
- package-setup
related: []
last_analyzed: '2026-03-09T07:44:10Z'
---

# fanuc_comms ROS 2 Package Setup

This document describes the standard Python setuptools setup.py for the fanuc_comms ROS 2 package, following the ament_python build convention. It declares the package name, version, and installs shared resources required by the ROS 2 ament index including resource marker files and package.xml. The configuration bundles a ROS 2 launch file (launch/fanuc_launch.py) into the package's share directory and registers a console script entry point mapping the CLI command 'fanuc_comms' to the main() function in fanuc_comms/fanuc_comms.py. The package is maintained by Wesley Havener at Saber Engineering and uses pytest for testing.

**Key concepts:** `ROS 2 ament_python build system`, `setuptools package configuration`, `console script entry point`, `ament resource index registration`, `launch file distribution`

## Sections

- **fanuc_comms ROS 2 Package Setup** — Overview of the Python setuptools configuration for the fanuc_comms ROS 2 ament package, including resource registration, launch file bundling, and console script entry point.
- **Exports** — Documents the console_scripts entry point that registers the fanuc_comms CLI command invoking main() from fanuc_comms/fanuc_comms.py.
- **Dependencies** — Lists setuptools as the package dependency required for the build configuration.
