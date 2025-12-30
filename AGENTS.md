# Agent Instructions for Cherry Processing Machine

## Build & Test Commands
- **Build all:** `colcon build --symlink-install`
- **Build package:** `colcon build --packages-select <pkg>`
- **Test all:** `colcon test`
- **Single test:** `colcon test --packages-select <pkg> --pytest-args -k <test_name>`
- **Check results:** `colcon test-result --all`

## Code Style Guidelines
- **Framework:** ROS2 Humble (`rclpy`, `rclcpp`). Use custom interfaces from `cherry_interfaces`.
- **Python:** PEP8 compliant. Use `snake_case` for functions/vars, `PascalCase` for classes.
- **C++:** `PascalCase` for classes/methods. Private members end with underscore (e.g., `nh_`).
- **Imports:** Group by (1) standard lib, (2) ROS2 modules, (3) local package modules.
- **Error Handling:** Wrap hardware/service calls in `try-except` (Python) or `try-catch` (C++). Use `self.get_logger().error()` or `RCLCPP_ERROR()` for logging.
- **Naming:** Use descriptive names for nodes and topics (e.g., `detector_node`, `~/detect`).
- **Coordinates:** Convert pixels to real-world meters using `Frame_tf` before tracking/actuation.
