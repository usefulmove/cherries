# Known Issues & Technical Debt

This document tracks bugs, architectural issues, and cleanup items to be addressed after the repository restructuring.

## Model Loading Bug (High Priority)

### Issue
The `cherry_detection` node loads model weights from the wrong package directory due to a configuration error in the source code.

### Location
- **File:** `cherry_system/cherry_detection/cherry_detection/detector.py:68`
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

## Code Duplication & Legacy Artifacts

### Issue
The `control_node` package contains a full copy of the detection code and model weights, which are legacy artifacts from before the detection logic was moved to its own package.

### Details
- **Duplicate Files:** `detector.py` and `ai_detector.py` exist in both `control_node` and `cherry_detection`.
- **Runtime Behavior:** `control_node` disables its local detector (line 159) and uses the `cherry_detection` service instead.
- **Result:** The code in `control_node` is "dead code", but the *model files* in `control_node` are active (due to the bug above).

---

## Open Task List (Cleanup)

1.  **Fix Model Loading Bug:**
    - Update `cherry_detection/detector.py` to use `get_package_share_directory('cherry_detection')`.

2.  **Verify & Remove Dead Code:**
    - Confirm `control_node` relies solely on the service call.
    - Remove `detector.py`, `ai_detector.py`, and model weights from `control_node`.

3.  **Clean Up Resources:**
    - Ensure `cherry_detection/resource/` contains the correct, active model weights.
    - Delete unused/duplicate weights to reclaim disk space.
