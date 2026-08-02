# Guiden Platform - Known Issues & Bug Resolution Report 🐛✅

> **Comprehensive audit report detailing resolved bugs, security patches, memory leak fixes, and code improvements across the Flutter mobile app and Python FastAPI server.**

---

## 📌 Issue Resolution Matrix

| ID | Issue Title | Target File | Assigned GitHub Labels | Status |
| :-: | :--- | :--- | :--- | :-: |
| **BUG-01** | Missing `HTTPException` Import in Server | [`guiden-server/assist_server.py`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L30) | `bug` | ✅ **Fixed** |
| **BUG-02** | Multi-plane YUV / NV12 Camera Stream Crash | [`lib/modules/guider/camera/guider_camera_controller.dart`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/guider/camera/guider_camera_controller.dart#L873) | `bug`, `help wanted` | ✅ **Fixed** |
| **SEC-01** | Exposed Replicate & ElevenLabs Private API Keys | [`guiden-server/assist_server.py`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L36) | `bug` | ✅ **Fixed** |
| **MEM-01** | Unbounded Base64 Session Memory Leak | [`guiden-server/assist_server.py`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L479) | `bug` | ✅ **Fixed** |
| **MEM-02** | Stream Listener Leak in `TtsService` | [`lib/services/tts_service.dart`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/services/tts_service.dart#L57) | `bug` | ✅ **Fixed** |
| **NET-01** | Hardcoded Local Wi-Fi IP Address (`172.20.10.2`) | [`lib/modules/guider/camera/guider_camera_controller.dart`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/guider/camera/guider_camera_controller.dart#L21) | `enhancement` | ✅ **Fixed** |
| **BUG-03** | Row Stride Padding Scan in Light Frequency | [`lib/modules/light-frequency/light_frequency_controller.dart`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/light-frequency/light_frequency_controller.dart#L125) | `bug` | ✅ **Fixed** |
| **BUG-04** | StateError on Empty `availableCameras()` | [`lib/main.dart`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/main.dart#L23) | `bug`, `good first issue` | ✅ **Fixed** |

---

## 🚨 Detailed Resolution Breakdown

### 1. BUG-01: Missing `HTTPException` Import (Python Server Crash)
* **Status**: ✅ **Fixed**
* **GitHub Label**: `bug`
* **Target File**: [`guiden-server/assist_server.py:L30`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L30)
* **Resolution**:
  Imported `HTTPException` from `fastapi` at line 30 so exception handlers in `/api/product/identify` and `/api/product/ask` return valid HTTP 500 JSON error responses without raising `NameError`.
  ```python
  from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
  ```

---

### 2. BUG-02: Camera Stream Crash on Dual-Plane YUV / iOS (`RangeError`)
* **Status**: ✅ **Fixed**
* **GitHub Labels**: `bug`, `help wanted`
* **Target File**: [`lib/modules/guider/camera/guider_camera_controller.dart:L873`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/guider/camera/guider_camera_controller.dart#L873)
* **Resolution**:
  Updated `_YuvArgs.from` factory to safely check `i.planes.length` before indexing plane 2, supporting 1-plane, 2-plane (NV12 / iOS), and 3-plane YUV formats without throwing `RangeError`.
  ```dart
  uBytes: i.planes.length > 1 ? i.planes[1].bytes : i.planes[0].bytes,
  vBytes: i.planes.length > 2 ? i.planes[2].bytes : (i.planes.length > 1 ? i.planes[1].bytes : i.planes[0].bytes),
  ```

---

### 3. SEC-01: Hardcoded Secret API Tokens Exposed
* **Status**: ✅ **Fixed**
* **GitHub Label**: `bug`
* **Target File**: [`guiden-server/assist_server.py:L36`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L36) & [`blind_navigation.py:L8`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/blind_navigation.py#L8)
* **Resolution**:
  Removed hardcoded secret tokens for Replicate and ElevenLabs. API keys are now strictly loaded from environment variables via `os.getenv(...)`.
  ```python
  REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
  ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
  ```

---

### 4. MEM-01: Unbounded Base64 Session Memory Leak
* **Status**: ✅ **Fixed**
* **GitHub Label**: `bug`
* **Target File**: [`guiden-server/assist_server.py:L479`](file:///C:/Users/ankus/Downloads/Guiden-AI/guiden-server/assist_server.py#L479)
* **Resolution**:
  Added `_clean_expired_product_sessions()` routine to purge product image payload entries older than 15 minutes (900 seconds) prior to inserting new session entries.

---

### 5. MEM-02: Stream Listener Leak in `TtsService`
* **Status**: ✅ **Fixed**
* **GitHub Label**: `bug`
* **Target File**: [`lib/services/tts_service.dart:L57`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/services/tts_service.dart#L57)
* **Resolution**:
  Added `StreamSubscription? _playerSub` management in `TtsService` to cancel stale audio player state stream listeners before registering a new listener and upon service disposal.

---

### 6. NET-01: Hardcoded Local Wi-Fi IP Address (`172.20.10.2`)
* **Status**: ✅ **Fixed**
* **GitHub Label**: `enhancement`
* **Target File**: [`lib/modules/guider/camera/guider_camera_controller.dart:L21`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/guider/camera/guider_camera_controller.dart#L21) & [`voice_assist_controller.dart:L32`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/voice-assist/voice_assist_controller.dart#L32)
* **Resolution**:
  Abstracted hardcoded IP addresses to read dynamically from `--dart-define` environment parameters (`WS_URL` and `SERVER_URL`) with safe defaults (`10.0.2.2`).

---

### 7. BUG-03: Row Stride Padding Scanning in Light Frequency Controller
* **Status**: ✅ **Fixed**
* **GitHub Label**: `bug`
* **Target File**: [`lib/modules/light-frequency/light_frequency_controller.dart:L125`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/modules/light-frequency/light_frequency_controller.dart#L125)
* **Resolution**:
  Changed inner column loop condition from `col < stride` to `col < image.width`, preventing inclusion of row alignment padding bytes in average illuminance calculations.

---

### 8. BUG-04: Unhandled Empty Camera List on Startup
* **Status**: ✅ **Fixed**
* **GitHub Labels**: `bug`, `good first issue`
* **Target File**: [`lib/main.dart:L23`](file:///C:/Users/ankus/Downloads/Guiden-AI/lib/main.dart#L23)
* **Resolution**:
  Wrapped `availableCameras()` in a try-catch block and defaulted to `<CameraDescription>[]` on error, preventing app startup crashes when camera permissions are not granted.
