# Known Issues & Technical Debt

This document tracks bugs, architectural issues, and cleanup items.

> **Note:** Issues related to the legacy `cherry_system` have been marked as OBSOLETE following its removal on 2026-02-17.

## Model Loading Bug (OBSOLETE - Legacy System)

### Issue
The `cherry_detection` node loads model weights from the wrong package directory due to a configuration error in the source code.

### Location
- **File:** `cherry_system/cherry_detection/cherry_detection/detector.py:68` (Removed)
- **Current code:**
  ```python
  package_share_directory = get_package_share_directory('control_node')  # BUG!
  ```
- **Should be:**
  ```python
  package_share_directory = get_package_share_directory('cherry_detection')
  ```

### Impact
- **Dead Code/Assets:** The model files in `cherry_detection/resource/` are never loaded, wasting ~259MB of disk space.
- **Maintenance Hazard:** Updating models in `cherry_detection/resource/` will have no effect on the running system.
- **Dependence on Legacy Artifacts:** The system relies on model files inside `control_node`, which technically shouldn't own them.

---

## Code Duplication & Legacy Artifacts (OBSOLETE - Resolved)

### Issue
The `control_node` package contains a full copy of the detection code and model weights.

### Status
Resolved by removal of `cherry_system`.

---

## Open Task List (Cleanup)

1.  **Fix Model Loading Bug:** (Done - System Replaced)
2.  **Verify & Remove Dead Code:** (Done - cherry_system removed)
3.  **Clean Up Resources:** (Done)
