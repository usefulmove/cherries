# Cherry Processing System Overview

ROS2 Humble-based industrial system for detecting, classifying, and tracking cherries on a conveyor belt, communicating coordinates to a FANUC robot arm for cherry sorting.

## Package-by-Package Breakdown

---

### 1. **cherry_interfaces** (Custom Message/Service/Action Definitions)

| File | Purpose | Used By |
|------|---------|----------|
| `msg/Cherry.msg` | Single cherry data (x, y, type) | All packages |
| `msg/CherryArray.msg` | Array of cherries | Fanuc, tracking |
| `msg/CherryArrayStamped.msg` | Timestamped cherry array | control_node, tracking |
| `msg/ImageSet.msg` | Paired top+bottom camera images with offsets | camera, control |
| `srv/Detection.srv` | Detect cherries in color image | cherry_detection |
| `srv/CombineImages.srv` | Merge top+bottom images | control_node |
| `srv/SetLights.srv` | Control top/bottom lights | usb_io |
| `srv/SetLatch.srv` | Set encoder latch state | usb_io |
| `srv/EncoderCount.srv` | Get current encoder count | usb_io |
| `action/Acquisition.action` | Asynchronous image acquisition | avt_camera, control |

---

### 2. **cherry_detection** (AI Detection Node)

| File | Purpose | Usage |
|------|---------|--------|
| `detector_node.py` | ROS2 service server for cherry detection | Provides `/detect` service |
| `detector.py` | Wrapper class for AI detection + coordinate transforms | Loads models, converts pixels→meters |
| `ai_detector.py` | PyTorch AI models (MaskRCNN + ResNet50) | Segments cherries, classifies as clean/pitted |

**Model Files:**
- `cherry_segmentation.pt` (176MB) - MaskRCNN instance segmentation model
- `cherry_classification.pt` (94MB) - ResNet50 binary classifier
**How Used:**
- Receives color image (RGB, with top camera in red, bottom in blue channel)
- Runs MaskRCNN to detect cherries
- Extracts cherry ROIs, classifies with ResNet50
- Converts pixel coordinates to real-world meters (rotation + scaling)
- Returns list of cherries (x, y, type) to caller

---

### 3. **control_node** (Main Orchestration Node)

| File | Purpose | Usage |
|------|---------|--------|
| `control_node.py` | Main ROS2 node - orchestrates entire pipeline | Central coordinator |
| `detector.py` | Detection wrapper (uses cherry_detection ai_detector) | Calls detection service |
| `ai_detector.py` | Duplicate of cherry_detection's AI module | Local detection capability |
| `frame_tracker.py` | Tracks multiple frames of cherries on conveyor | Updates cherry positions via encoder |
| `frame_tf.py` | Represents a frame with cherry detections | Manages offset calculations |
| `cherry_types.py` | Enum: PIT, NO_PIT, UNKNOWN | Cherry type definitions |
| `point.py` | Simple Point class (x, y) | Coordinate utilities |
| `cherry_tf.py` | Cherry class with type/location | Appears unused |
| `Frame.py` | Old frame-based acquisition node | Appears unused |

**How Used:**
1. Listens to `usb_io_node/encoder_change` for conveyor movement
2. Every 500 encoder ticks → triggers new frame acquisition
3. Calls `single_acquisition_server` action → gets top+bottom images
4. Combines images into RGB composite (top=red, bottom=blue)
5. Calls `detection_server/detect` service → gets cherry detections
6. Adds detections to frame tracker (with position offsets)
7. Publishes `control_node/cherries` (CherryArrayStamped) at 60 Hz for projectors
8. Optionally saves images (via `enable_image_save` parameter)

---

### 4. **usb_io** (Hardware I/O Node)

| File | Purpose | Usage |
|------|---------|--------|
| `usb_io_node.py` | DAQ device interface for encoder + digital I/O | Reads conveyor encoder, controls lights |

**How Used:**
- **Encoder Reading:** Uses uldaq library to read 32-bit encoder counter at 10kHz sample rate
- **Topics Published:**
  - `~/encoder_count` - Current absolute encoder count
  - `~/encoder_change` - Delta changes (used by control_node)
  - `~/encoder_speed` - Conveyor speed (cherries/sec)
- **Services Provided:**
  - `~/get_count` - Get current encoder value
  - `~/set_lights` - Toggle top/bottom lights
  - `~/set_encoder_latch` - Set encoder latch bit
- **Topics Subscribed:**
  - `~/top_light`, `~/back_light`, `~/encoder_latch` - Control signals
- **Digital Output:** Controls 8-bit port for lights + encoder latch (relay outputs)
- **Parameters:** `simulate_encoder` - For testing without hardware

---

### 5. **fanuc_comms** (FANUC Robot Communication)

| File | Purpose | Usage |
|------|---------|--------|
| `fanuc_comms.py` | TCP/IP server for FANUC robot | Sends cherry coordinates to robot |

**How Used:**
- Listens on TCP port 59002 (IP: 10.0.0.10)
- Receives trigger command: `RUNFIND\r` or `RUNFIND,<id>\r`
- Responds: `OK\r` on success, `ER\r` on error
- Flow:
  1. Receive trigger → Set encoder latch true
  2. Call `single_acquisition_server` → Get images
  3. Combine images → color RGB
  4. Call `detection_server/detect` → Get cherries
  5. Filter cherries (exclude type 1/clean, keep type 2/pitted)
  6. Convert to mm, offset by (-155, -457) for robot frame
  7. Send back: `1,<count>,x1,y1,x2,y2,...\r` (max 7 points)
  8. Set encoder latch false

---

### 6. **tracking_projector** (Visualization/Projection)

| File | Purpose | Usage |
|------|---------|--------|
| `src/main.cpp` | Qt application entry point | ROS2 + Qt integration |
| `src/tracker.cpp` | Main tracker widget (QOpenGLWidget) | Subscribes to cherries, renders |
| `src/helper.cpp` | Painting utilities (circles, grid) | Draws cherries by type |
| `src/cherry_cpp.cpp` | Simple C++ Cherry class | Data structure |
| `include/tracking_projector/*.h*` | Headers | Interface definitions |
| `include/quaternion/*.h` | Quaternion utilities | Possibly for 3D transforms |

**How Used:**
- Subscribes to `control_node/cherries` (CherryArrayStamped)
- Transforms cherry coordinates (translation + rotation + scaling) to screen space
- Renders at 60 FPS using QPainter/QOpenGLWidget
- **Cherry Types Rendered:**
  - Type 2 (pitted) → Magenta circles
  - Type 3 (side) → Cyan circles  
  - Type 5 (maybe) → Yellow circles
  - Type 1 (clean) → NOT rendered (filtered out)
- **Parameters:**
  - `x`, `y` - Projector offset in meters
  - `scaling_factor` - Pixels per meter
  - `rotation` - Screen rotation (radians)
  - `screen` - Display index (0, 1, 2...)
  - `show_grid` - Overlay calibration grid
  - `circle_size` - Circle diameter (mm)
**Launch:**
- Two instances typically launched (`projector_1`, `projector_2`) with different screen positions

---

### 7. **avt_vimba_camera** (Allied Vision Camera Driver)

| File | Purpose | Usage |
|------|---------|--------|
| `src/single_acquisition_node.cpp` | Action server for image acquisition | Provides `single_acquisition_server` |
| `src/mono_camera_node.cpp` | Continuous camera streaming | (Alternative mode) |
| `src/avt_vimba_camera.cpp` | Core Vimba API interface | Camera management |
| `src/ApiController.cpp` | Camera API controller | Low-level control |
| `src/frame_observer.cpp` | Frame callback handler | Image buffer management |
| `src/trigger_node.cpp` | Hardware trigger management | (Alternative trigger source) |

**How Used:**
- Connects to AVT GigE camera at IP 172.16.1.2
- **Action Server:** `single_acquisition_server`
- **Action Flow:**
  1. Receive goal with `frame_id`
  2. Wait for encoder latch to go true (trigger condition)
  3. Strobe top light → capture top image
  4. Strobe back light → capture back image
  5. Return `ImageSet` (top + back images with timing offsets)
- **Topics Subscribed:**
  - `~/encoder_change` - Update camera position
  - `/usb_io_node/top_light`, `/usb_io_node/back_light` - Light control
- **Parameters:**
  - `acquisition_delay` - Trigger delay (μs)
  - `top_light_gain`, `back_light_gain` - Camera exposure
  - `frame_id` - Frame identifier

---

### 8. **camera_simulator** (Test/Simulation)

| File | Purpose | Usage |
|------|---------|--------|
| `camera_simulator.py` | Mock camera action server | Development/testing |

**How Used:**
- Implements same `single_acquisition_server` action as real camera
- Returns static test images from `/home/user/Pictures/`
- Uses random offsets for testing timing behavior

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Conveyor Movement                              │
│              usb_io_node → encoder_change (Int64)                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   control_node    │
                    │  (orchestrator)   │
                    └─────────┬─────────┘
                              │
                              ▼ (every 500 ticks)
        ┌────────────────────────────────────────┐
        │  single_acquisition_server (Action)    │
        │  (avt_vimba_camera)                    │
        └───────────────────┬────────────────────┘
                            │
                            ▼
                 ┌───────────────────┐
                 │  combine_images() │
                 │  (RGB merge)      │
                 └─────────┬─────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  detection_server/detect │
              │  (cherry_detection)      │
              └───────────┬──────────────┘
                          │
                          ▼
                  ┌───────────────────┐
                  │  frame_tracker    │
                  │  (update pos)     │
                  └─────────┬─────────┘
                            │
                            ▼
          ┌─────────────────────────────┐
          │  control_node/cherries      │
          │  (CherryArrayStamped @60Hz) │
          └───────────┬─────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
┌─────────────────┐            ┌─────────────────┐
│ tracking_proj   │            │   fanuc_comms   │
│ (visualization) │            │ (robot control) │
└─────────────────┘            └─────────────────┘
```

## Cherry Classification Types

| Code | Label | Description | Action |
|------|-------|-------------|---------|
| 0 | None | Background | Ignore |
| 1 | Clean cherry | Good cherry | Pass through (not sent to robot) |
| 2 | Pitted cherry | Defective | Send to robot for sorting |
| 3 | Side | Cherry on edge of conveyor | Track but don't sort |
| 5 | Maybe | Uncertain classification | Yellow visual marker |

## Launch Configurations

| Launch File | Purpose | Extra Nodes |
|-------------|---------|--------------|
| `conveyor_control_launch.py` | Full conveyor system | control_node |
| `fanuc_launch.py` | FANUC robot integration | fanuc_comms |

Both launch:
- `cherry_detection` (detection service)
- `avt_vimba_camera` (camera)
- `usb_io` (encoder + lights)
- 2x `tracking_projector` (dual projectors) with position parameters
