---
name: Tracking & Orchestration
layer: Orchestration
impact_area: System Logic, Robot Control, UX
---

# Tracking & Orchestration Layer

## Responsibility
The central nervous system. It coordinates the timing of all other layers, maintains the "World Model" of where every cherry is on the moving belt, and handles external communication.

## Key Components

### 1. Control Node (`cherry_system/control_node/`)
*   **Role**: The conductor.
*   **Logic**:
    *   **Triggering**: Monitors `usb_io` encoder ticks. Fires `avt_vimba_camera`.
    *   **Image Assembly**: Merges Top/Bottom images.
    *   **Inference Request**: Calls `cherry_detection` service.
    *   **Frame Tracking**: `FrameTracker` class maintains a queue of detected cherries, updating their position ($x, y$) based on the moving belt ($\Delta \text{ticks}$).
    *   **Decision**: Decides when a cherry reaches the ejector or robot pick zone.

### 2. Robot Communication (`cherry_system/fanuc_comms/`)
*   **Protocol**: TCP/IP socket communication with Fanuc robot controller.
*   **Data**: Sends $(x, y, \text{class})$ coordinates for sorting.

### 3. Visualization (`cherry_system/tracking_projector/`)
*   **Tech**: C++ / Qt / OpenGL.
*   **Function**: Projects colored circles directly onto the physical cherries on the belt for real-time human feedback and manual review.

**Color Coding by Classification:**

| Classification Label | Color | Purpose |
|:---------------------|:------|:--------|
| 1 (Clean) | **Green** | Pass—cherry proceeds to good bin |
| 2 (Pit) | **Red** | Fail—cherry is rejected or sorted separately |
| 3 (Side) | **Cyan** | Edge case—cherry at image boundary |
| 5 (Maybe) | **Yellow** | **Manual review required**—worker inspects uncertain prediction |

**Key Implementation:**
- **Brush definitions** (`helper.cpp:66-69`): `circleBrush_maybe = QBrush(Qt::yellow)`
- **Rendering** (`helper.cpp:115-134`): Type 5 cherries rendered with yellow brush
- **Safety Workflow**: Yellow highlights enable human oversight of model uncertainty, preventing automatic misclassification of ambiguous cases

*   **Sync**: Subscribes to the tracker state to keep projections locked to the moving fruit.

## Design Decisions
*   **Service-Based Inference**: Inference is blocking and heavy. `control_node` offloads it to `cherry_detection` (Service) so the main control loop (Encoder counting) never blocks.

## Discovery Links
*   **Code**: `src/cherry_system/control_node/`
*   **Visualization Code**: `src/cherry_system/tracking_projector/`
*   **Classification Categories**: See [Inference Pipeline](../inference_pipeline/ARCHITECTURE.md) for threshold logic and category definitions
*   **Projection Rendering**: `cherry_system/tracking_projector/src/helper.cpp:66-69` (color brushes), `115-134` (rendering logic)
