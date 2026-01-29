# Cherry Processing Standards

## Coding Conventions

### Python
- **Style**: Adhere to PEP8.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Logging**: Use ROS2 node loggers (`self.get_logger()`).
- **Imports**: Group by standard library, ROS2 modules, and local packages.
- **Error Handling**: Wrap hardware and service calls in `try-except` blocks.

### C++
- **Style**: ROS2 C++ coding style.
- **Naming**: `PascalCase` for classes/methods; private members end with underscore (`member_`).
- **Logging**: Use `RCLCPP_*` macros.

## ROS2 Patterns
- **Interfaces**: All custom communication must use `cherry_interfaces`.
- **Parameters**: Use the ROS2 parameter system for configurable thresholds and paths.
- **Lifecycle**: Use Actions for long-running tasks (Acquisition) and Services for request-response (Detection).

## ML & Training Workflow
- **Environment**: Use PyTorch with `torchvision`.
- **Configuration**: Use YAML files for experiment hyperparameters.
- **Sync**: Maintain training scripts locally; execute on Colab via Google Drive staging (Option 2).
- **Versioning**: Git tracks code and configs; Google Drive tracks model weights (`.pt`) and datasets.
- **Exclusions**: Large binary files (`.pt`) must be kept out of the Git repository via `.gitignore`.
