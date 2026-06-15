# docs/system_architecture/cameras/src/

Directory containing 3 analyzed file(s).

## Files

- [EventObserver.md](EventObserver.md): Implements the EventObserver class, a concrete VmbCPP::IFeatureObserver that forwards camera feature-change notifications to a user-supplied callback function. — `camera`, `vmbcpp`, `observer-pattern`
- [INDEX.md](INDEX.md): Index of C++ source implementations for ROS 2 camera driver nodes targeting the Allied Vision/Cognex CIC5000 GigE camera via the VmbCPP SDK. — `camera-driver`, `ros2`, `vmbcpp`
- [cognex_hdr.md](cognex_hdr.md): Implements a ROS2 node (Cic5000CameraHdr) that drives an Allied Vision GigE camera via the VmbCPP SDK to capture HDR image sequences with configurable gain/exposure per frame, publishing results as image sets and supporting both action-server and topic-based triggering. — `ros2`, `camera-driver`, `hdr-imaging`
