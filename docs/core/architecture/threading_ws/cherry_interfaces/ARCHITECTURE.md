---
name: cherry_interfaces
layer: Interface
impact_area: All Packages
---

# cherry_interfaces Package

**Location:** `threading_ws/src/cherry_interfaces/`  
**Type:** ROS2 Interface Package (msg/srv/action definitions)  
**Status:** Production - HDR-capable, expanded from legacy

## Purpose

Defines the communication contract for all cherry processing system nodes. This is an **expanded interface** (16 messages, 14 services, 3 actions) compared to the legacy cherry_system (4 messages, 6 services, 2 actions).

## Message Types

### Core Cherry Data

| Message | Fields | Purpose | Usage |
|:--------|:-------|:--------|:------|
| **Cherry.msg** | `float32 x, y`, `bytes type`, `float32 confidence` | Single cherry detection | Detection results, robot targets |
| **CherryArray.msg** | `Cherry[] cherries` | Array of detections | Service responses, publications |
| **CherryArrayStamped.msg** | `std_msgs/Header header`, `CherryArray cherries` | Stamped array | Time-synchronized publications |

### Image Data (Legacy Simple)

| Message | Fields | Purpose | Usage |
|:--------|:-------|:--------|:------|
| **ImageSet.msg** | `sensor_msgs/Image[] images`, `int64[] counts` | Simple image collection | Legacy pipeline, basic detection |

### Image Data (HDR Multi-Layer)

| Message | Fields | Purpose | Usage |
|:--------|:-------|:--------|:------|
| **ImageLayer.msg** | `string name`, `string frame_id`, `sensor_msgs/Image image`, `int64 count`, `int64 mm` | Single layer with metadata | Multi-layer HDR processing |
| **ImageSetHdr.msg** | `Trigger trigger`, `ImageLayer[] images` | HDR image set | Production detection pipeline |

### System State

| Message | Fields | Purpose | Usage |
|:--------|:-------|:--------|:------|
| **Trigger.msg** | `builtin_interfaces/Time stamp`, `uint64 frame_id`, `int64 encoder_count`, `int64 encoder_mm` | Acquisition trigger | Camera triggering, frame coordination |
| **EncoderCount.msg** | `int64 count`, `int64 mm` | Encoder position | Position tracking, synchronization |
| **PickMode.msg** | `bool automatic`, `bool pick_pits`, `bool pick_clean`, etc. | Robot mode settings | Fanuc configuration |
| **Inputs.msg** | `bool[] inputs` | Digital inputs | PLC I/O status |
| **Outputs.msg** | `bool[] outputs` | Digital outputs | PLC I/O control |
| **HSC.msg** | `int64 count` | High-speed counter | Encoder interface |
| **Temperature.msg** | `float32 temperature` | System temperature | Monitoring |

## Service Types

### Detection Services

| Service | Request | Response | Purpose |
|:--------|:--------|:---------|:--------|
| **Detection.srv** | `sensor_msgs/Image image` | `Cherry[] cherries`, `sensor_msgs/Image keypoint_image` | Legacy single-image detection |
| **Detectionhdr.srv** | `uint64 frame_id`, `sensor_msgs/Image image_top1-3, image_bot1-3`, `int64 count_top1-3, count_bot1-3`, `int64 mm_top1-3, mm_bot1-3` | `sensor_msgs/Image keypoint_image`, `Cherry[] cherries`, `int64 encoder_count` | HDR multi-layer detection |

### Image Services

| Service | Request | Response | Purpose |
|:--------|:--------|:---------|:--------|
| **CombineImages.srv** | `ImageSet images` | `sensor_msgs/Image combined` | Layer combination |
| **ImageSave.srv** | `sensor_msgs/Image image`, `string filename` | `bool success` | Image persistence |
| **ImageCounts.srv** | (empty) | `int64 count` | Image counter query |

### Encoder/PLC Services

| Service | Request | Response | Purpose |
|:--------|:--------|:---------|:------|
| **EncoderCount.srv** | (empty) | `int64 count`, `int64 mm` | Get current encoder position |
| **EncoderLatches.srv** | (empty) | `int64[] counts` | Get latched encoder values |
| **SetLatch.srv** | `int32 index`, `bool enable` | `bool success` | Configure encoder latch |
| **ResetLatches.srv** | (empty) | `bool success` | Clear all latches |

### Robot Services

| Service | Request | Response | Purpose |
|:--------|:--------|:---------|:--------|
| **LatchRobot.srv** | (empty) | `bool latched` | Get robot latch status |
| **GetCherryBuffer.srv** | (empty) | `CherryArray cherries` | Get buffered cherries for robot |

### System Services

| Service | Request | Response | Purpose |
|:--------|:--------|:---------|:--------|
| **SetLights.srv** | `bool top_light`, `bool bot_light` | `bool success` | Control illumination |
| **Trigger.srv** | `builtin_interfaces/Time stamp` | `bool success` | Manual trigger |

## Action Types

| Action | Goal | Feedback | Result | Purpose |
|:-------|:-----|:---------|:-------|:--------|
| **Acquisition.action** | `std_msgs/Header header` | `int32 status` | `ImageSet images`, `int64[] counts` | Legacy image acquisition |
| **Acquisitionhdr.action** | `std_msgs/Header header` | `int32 status`, `string message` | `ImageSetHdr image_set` | HDR image acquisition |
| **FindCherries.action** | `std_msgs/Header header` | `int32 status`, `CherryArray cherries` | `CherryArray cherries`, `int64 encoder_count` | Complete detect-and-track |

## Usage Patterns

### Production HDR Pipeline

```
composite (FindCherries.action)
    ↓
trigger_node (Trigger.msg)
    ↓
cameras (Acquisitionhdr.action)
    ↓
cherry_detection (Detectionhdr.srv)
    ↓
cherry_buffer (GetCherryBuffer.srv)
    ↓
fanuc_comms (CherryArray.msg)
```

### Legacy Simple Pipeline

```
control_node
    ↓
camera_simulator (single image)
    ↓
cherry_detection (Detection.srv)
    ↓
fanuc_comms (CherryArrayStamped.msg)
```

## Migration Notes

### New in threading_ws (Not in cherry_system)

**Messages Added:**
- ImageLayer.msg
- ImageSetHdr.msg
- Trigger.msg
- EncoderCount.msg
- HSC.msg
- PickMode.msg
- Inputs.msg
- Outputs.msg
- Temperature.msg

**Services Added:**
- Detectionhdr.srv
- EncoderLatches.srv
- ResetLatches.srv
- GetCherryBuffer.srv
- LatchRobot.srv
- ImageCounts.srv
- Trigger.srv

**Actions Added:**
- Acquisitionhdr.action
- FindCherries.action

### Breaks cherry_system Compatibility

Services with incompatible changes:
- Detection.srv vs Detectionhdr.srv (different request/response)
- Acquisition.action vs Acquisitionhdr.action (different result type)

## Code Examples

### Publishing CherryArray

```python
from cherry_interfaces.msg import Cherry, CherryArray

cherry = Cherry()
cherry.x = 0.5
cherry.y = 0.3
cherry.type = (1).to_bytes(1, 'big')  # Type 1 = clean

array = CherryArray()
array.cherries = [cherry]
publisher.publish(array)
```

### Using ImageSetHdr

```python
from cherry_interfaces.msg import ImageSetHdr, ImageLayer, Trigger

trigger = Trigger()
trigger.frame_id = 12345
trigger.encoder_count = 1000000

layer = ImageLayer()
layer.name = 'top2'
layer.image = image_msg
layer.count = 1000000

hdr_set = ImageSetHdr()
hdr_set.trigger = trigger
hdr_set.images = [layer]
```

### Calling Detectionhdr

```python
from cherry_interfaces.srv import Detectionhdr

client = self.create_client(Detectionhdr, '/cherry_detection/detect')
request = Detectionhdr.Request()
request.frame_id = 12345
request.image_top2 = top_image
request.image_bot1 = bot1_image
request.image_bot2 = bot2_image
request.count_bot1 = encoder_count1
request.count_bot2 = encoder_count2
# ... set other fields

future = client.call_async(request)
```

## Related Documentation

- [threading_ws INDEX](../INDEX.md)
- [cherry_detection Architecture](./cherry_detection/ARCHITECTURE.md)
- [composite Architecture](./composite/ARCHITECTURE.md)
- [Migration Guide](../../reference/MIGRATION_cherry_system_to_threading_ws.md)

## Discovery Links

- **Message Definitions:** `threading_ws/src/cherry_interfaces/msg/`
- **Service Definitions:** `threading_ws/src/cherry_interfaces/srv/`
- **Action Definitions:** `threading_ws/src/cherry_interfaces/action/`
- **CMakeLists.txt:** `threading_ws/src/cherry_interfaces/CMakeLists.txt` (see all interfaces listed)
