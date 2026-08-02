// guiden_camera_controller.dart
import 'dart:async';
import 'dart:convert';
import 'dart:isolate';

import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:image/image.dart' as img;
import 'package:speech_to_text/speech_to_text.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../../../../main.dart';
import '../../../services/hand_detector.dart';
import '../../../services/voice_assistant_controller.dart';
import '../../../services/yolo_model_manager.dart';

const _kWsUrl = String.fromEnvironment('WS_URL', defaultValue: 'ws://10.0.2.2:8765/ws/assist');
const _kUserId = 'user_001'; // persist this (SharedPrefs/UUID)

// ─── Models ───────────────────────────────────────────────────────────────────

class BBox {
  final String label;
  final double confidence;
  final double left, top, right, bottom;
  const BBox({
    required this.label,
    required this.confidence,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });
}

enum AssistStatus { idle, listening, thinking, speaking, stopped, error }

enum NavMode { idle, freeRoam, goalDirected }

// ─── Controller ───────────────────────────────────────────────────────────────

class MergedDetectionController extends GetxController {
  final taskSteps = <String>[].obs;
  final currentStepIndex = 0.obs;
  final taskProgress = 0.0.obs;

  // ── YOLO ──────────────────────────────────────────────────────────────────
  final detectionCount = 0.obs;
  final currentFps = 0.0.obs;
  final yoloBoxes = <BBox>[].obs;

  // ── Hand ──────────────────────────────────────────────────────────────────
  final handLandmarks = <Hand>[].obs;
  final currentGesture = HandGesture.unknown.obs;
  final isHandDetectionEnabled = true.obs;

  // ── Thresholds ────────────────────────────────────────────────────────────
  final confidenceThreshold = 0.25.obs;
  final iouThreshold = 0.45.obs;

  // ── Model / camera ────────────────────────────────────────────────────────
  final isModelLoading = true.obs;
  final loadingMessage = 'Initializing…'.obs;
  final downloadProgress = 0.0.obs;
  final isCameraReady = false.obs;
  final isFrontCamera = false.obs;
  final currentZoomLevel = 1.0.obs;

  // ── Assist ────────────────────────────────────────────────────────────────
  final assistStatus = AssistStatus.idle.obs;
  final assistTranscript = ''.obs;
  final assistResponse = ''.obs;
  final navMode = NavMode.freeRoam.obs;
  final currentGoal = ''.obs;
  final urgencyLevel = 'low'.obs; // for UI pulse effect

  CameraController? get cameraController => _camCtrl;

  // ─── Private ──────────────────────────────────────────────────────────────
  CameraController? _camCtrl;
  YOLO? _yolo;
  HandLandmarkerPlugin? _handPlugin;
  String? _modelPath;

  bool _isProcessingYolo = false;
  bool _isProcessingHand = false;
  int _frameCounter = 0;

  static const _yoloEveryN = 2;
  int _sensorOrientation = 0;

  DateTime _lastFpsTime = DateTime.now();
  int _fpsFrameCount = 0;

  final List<HandGesture> _gestureHistory = [];
  static const _historySize = 5;

  late final YoloModelManager _modelManager;

  // Isolate
  Isolate? _workerIsolate;
  SendPort? _workerSendPort;
  ReceivePort? _workerReceivePort;
  Completer<Uint8List?>? _yoloTaskCompleter;

  // Latest frame
  CameraImage? _latestFrame;

  // STT
  final SpeechToText _stt = SpeechToText();
  bool _sttAvailable = false;

  // WebSocket  – persistent for the whole session
  WebSocketChannel? _ws;
  StreamSubscription? _wsSub;
  bool _wsConnected = false;

  // Frame sending
  Timer? _frameSendTimer;
  bool _sendingFrame = false;

  // Audio
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _audioPlaying = false;

  // Gesture debounce
  HandGesture _lastSentGesture = HandGesture.unknown;
  DateTime _lastGestureSent = DateTime.now();
  DateTime _lastFrameSendTime = DateTime.fromMillisecondsSinceEpoch(0);

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  @override
  void onInit() {
    super.onInit();
    _modelManager = YoloModelManager(
      onDownloadProgress: (p) => downloadProgress.value = p,
      onStatusUpdate: (m) => loadingMessage.value = m,
    );
    _boot();

    _audioPlayer.onPlayerComplete.listen((_) {
      _audioPlaying = false;
      if (assistStatus.value == AssistStatus.speaking) {
        assistStatus.value = AssistStatus.idle;
      }
    });

    // Pause global voice assistant
    try {
      if (Get.isRegistered<VoiceAssistantController>()) {
        Get.find<VoiceAssistantController>().pauseVoiceAssistant();
      }
    } catch (_) {}
  }

  @override
  void onClose() {
    _frameSendTimer?.cancel();
    _camCtrl?.stopImageStream().catchError((_) {});
    _camCtrl?.dispose();
    _handPlugin?.dispose();
    _yolo?.dispose();
    _workerIsolate?.kill(priority: Isolate.immediate);
    _workerReceivePort?.close();
    _disconnectWs();
    _audioPlayer.dispose();

    // Resume global voice assistant
    try {
      if (Get.isRegistered<VoiceAssistantController>()) {
        Get.find<VoiceAssistantController>().resumeVoiceAssistant();
      }
    } catch (_) {}

    super.onClose();
  }

  // ─── Boot ─────────────────────────────────────────────────────────────────

  Future<void> _boot() async {
    isModelLoading.value = true;

    // STT
    _sttAvailable = await _stt.initialize(
      onStatus: (s) => debugPrint('[STT] $s'),
      onError: (e) => debugPrint('[STT] error: $e'),
    );

    // YOLO model
    _modelPath = await _modelManager.getModelPath();
    if (_modelPath == null) {
      loadingMessage.value = 'Failed to load model';
      isModelLoading.value = false;
      return;
    }

    // YOLO
    try {
      _yolo = YOLO(modelPath: _modelPath!, task: YOLOTask.detect, useGpu: true);
      await _yolo!.loadModel();
    } catch (e) {
      loadingMessage.value = 'YOLO error: $e';
      isModelLoading.value = false;
      return;
    }

    // Hand
    try {
      _handPlugin = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        delegate: HandLandmarkerDelegate.gpu,
      );
    } catch (e) {
      debugPrint('[Merged] HandLandmarker: $e');
    }

    // Worker isolate
    await _startWorkerIsolate();

    // Camera
    await _startCamera();

    // Connect WebSocket (persistent)
    await _connectWs();

    // Start sending frames every 600ms
    _startFrameTimer();

    loadingMessage.value = '';
    isModelLoading.value = false;
  }

  // ─── WebSocket (Persistent) ───────────────────────────────────────────────

  Future<void> _connectWs() async {
    try {
      _ws = WebSocketChannel.connect(Uri.parse(_kWsUrl));
      await _ws!.ready;
      _wsConnected = true;
      debugPrint('[WS] Connected');

      // Send init/handshake
      _wsSend({'type': 'init', 'user_id': _kUserId});

      _wsSub = _ws!.stream.listen(
        _onWsMessage,
        onDone: _onWsDone,
        onError: _onWsError,
      );
    } catch (e) {
      debugPrint('[WS] Connect error: $e');
      _wsConnected = false;
      // Retry after 3 seconds
      Future.delayed(const Duration(seconds: 3), _connectWs);
    }
  }

  void _onWsMessage(dynamic raw) {
    try {
      final msg = jsonDecode(raw as String) as Map<String, dynamic>;
      final type = msg['type'] as String? ?? '';

      switch (type) {
        case 'response':
          final text = msg['text'] as String? ?? '';
          final urgency = msg['urgency'] as String? ?? 'low';
          final mode = msg['mode'] as String? ?? 'free_roam';
          final goal = msg['goal'] as String? ?? '';

          // Task fields
          final steps = (msg['task_steps'] as List?)?.cast<String>() ?? [];
          final stepIdx = msg['task_step_idx'] as int? ?? 0;

          assistResponse.value = text;
          urgencyLevel.value = urgency;
          currentGoal.value = goal;

          if (steps.isNotEmpty) {
            taskSteps.value = steps;
            currentStepIndex.value = stepIdx;
            taskProgress.value = steps.isEmpty ? 0 : stepIdx / steps.length;
          }

          navMode.value = switch (mode) {
            'goal_directed' => NavMode.goalDirected,
            _ => NavMode.freeRoam,
          };

          if (text.isNotEmpty) {
            assistStatus.value = AssistStatus.thinking;
          }

        case 'audio':
          final audio64 = msg['audio'] as String? ?? '';
          final urgency = msg['urgency'] as String? ?? 'low';
          if (audio64.isNotEmpty) {
            _playAudio(
              audio64,
              interrupt: urgency == 'critical' || urgency == 'task_step',
            );
          }

        case 'task_complete':
          currentGoal.value = '';
          taskSteps.clear();
          currentStepIndex.value = 0;
          taskProgress.value = 1.0;
          debugPrint('[Task] Complete!');

        case 'status':
          debugPrint('[WS] Status: ${msg['message']}');

        case 'error':
          assistStatus.value = AssistStatus.error;
          Future.delayed(const Duration(seconds: 2), () {
            if (assistStatus.value == AssistStatus.error) {
              assistStatus.value = AssistStatus.idle;
            }
          });
      }
    } catch (e) {
      debugPrint('[WS] Parse error: $e');
    }
  }

  void _onWsDone() {
    debugPrint('[WS] Disconnected');
    _wsConnected = false;
    // Reconnect
    Future.delayed(const Duration(seconds: 2), _connectWs);
  }

  void _onWsError(Object e) {
    debugPrint('[WS] Error: $e');
    _wsConnected = false;
  }

  void _disconnectWs() {
    _wsSub?.cancel();
    _ws?.sink.close();
    _wsConnected = false;
  }

  void _wsSend(Map<String, dynamic> data) {
    if (_wsConnected) {
      try {
        _ws!.sink.add(jsonEncode(data));
      } catch (e) {
        debugPrint('[WS] Send error: $e');
      }
    }
  }

  // ─── Frame Timer ──────────────────────────────────────────────────────────

  void _startFrameTimer() {
    _frameSendTimer?.cancel();
    // Check every 200ms, then the _sendLatestFrame logic will enforce 
    // the actual 600ms or 2000ms delay.
    _frameSendTimer = Timer.periodic(
      const Duration(milliseconds: 200),
      (_) => _sendLatestFrame(),
    );
  }

  Future<void> _sendLatestFrame() async {
    if (!_wsConnected || _sendingFrame) return;
    
    final now = DateTime.now();
    final isTaskMode = navMode.value == NavMode.goalDirected || taskSteps.isNotEmpty;
    final delayMs = isTaskMode ? 2000 : 600;
    
    if (now.difference(_lastFrameSendTime).inMilliseconds < delayMs) return;

    final frame = _latestFrame;
    if (frame == null) return;

    _sendingFrame = true;
    try {
      final jpeg = await compute(
        _processYuvToJpeg,
        _YuvArgs.from(frame, _sensorOrientation),
      );
      if (jpeg != null) {
        _wsSend({'type': 'frame', 'jpeg': base64Encode(jpeg)});
        _lastFrameSendTime = DateTime.now();
      }
    } catch (e) {
      debugPrint('[Frame] Error: $e');
    } finally {
      _sendingFrame = false;
    }
  }

  // ─── Audio ────────────────────────────────────────────────────────────────

  Future<void> _playAudio(String base64Audio, {bool interrupt = false}) async {
    if (_audioPlaying && !interrupt) return;
    if (interrupt) await _audioPlayer.stop();

    try {
      final bytes = base64Decode(base64Audio);
      _audioPlaying = true;
      assistStatus.value = AssistStatus.speaking;
      await _audioPlayer.play(BytesSource(bytes));
    } catch (e) {
      debugPrint('[Audio] Error: $e');
      _audioPlaying = false;
      assistStatus.value = AssistStatus.idle;
    }
  }

  // ─── Gesture → Server ─────────────────────────────────────────────────────

  void _onGestureChanged(HandGesture gesture) {
    // Debounce: only send if gesture changed and 800ms elapsed
    final now = DateTime.now();
    if (gesture == _lastSentGesture) return;
    if (now.difference(_lastGestureSent).inMilliseconds < 800) return;

    _lastSentGesture = gesture;
    _lastGestureSent = now;

    switch (gesture) {
      case HandGesture.fist:
        // Stop everything
        _audioPlayer.stop();
        _stt.stop();
        assistStatus.value = AssistStatus.stopped;
        _wsSend({'type': 'gesture', 'gesture': 'fist'});
        debugPrint('[Gesture] FIST - STOP');

      case HandGesture.openHand:
        // Palm = start listening
        if (assistStatus.value == AssistStatus.idle ||
            assistStatus.value == AssistStatus.stopped) {
          _startListening();
          _wsSend({'type': 'gesture', 'gesture': 'palm'});
          debugPrint('[Gesture] PALM - LISTEN');
        }

      case HandGesture.peace:
        // Peace = clear goal + resume
        _wsSend({'type': 'gesture', 'gesture': 'peace'});
        currentGoal.value = '';
        taskSteps.clear();
        currentStepIndex.value = 0;
        taskProgress.value = 0.0;
        navMode.value = NavMode.freeRoam;
        if (assistStatus.value == AssistStatus.stopped) {
          assistStatus.value = AssistStatus.idle;
        }
        debugPrint('[Gesture] PEACE - CLEAR GOAL');

      default:
        break;
    }
  }

  // ─── STT ──────────────────────────────────────────────────────────────────

  Future<void> _startListening() async {
    if (!_sttAvailable) return;
    if (assistStatus.value == AssistStatus.listening) return;

    assistStatus.value = AssistStatus.listening;
    assistTranscript.value = '';

    await _stt.listen(
      onResult: (result) {
        assistTranscript.value = result.recognizedWords;
        if (result.finalResult && result.recognizedWords.trim().isNotEmpty) {
          _stt.stop();
          _onQuestionReady(result.recognizedWords.trim());
        }
      },
      listenFor: const Duration(seconds: 10),
      pauseFor: const Duration(seconds: 2),
      localeId: 'en-US',
      listenOptions: SpeechListenOptions(partialResults: true),
    );

    _stt.statusListener = (status) {
      if ((status == 'done' || status == 'notListening') &&
          assistStatus.value == AssistStatus.listening) {
        final t = assistTranscript.value.trim();
        if (t.isNotEmpty) {
          _onQuestionReady(t);
        } else {
          assistStatus.value = AssistStatus.idle;
        }
      }
    };
  }

  void _onQuestionReady(String question) {
    assistStatus.value = AssistStatus.thinking;
    debugPrint('[STT] Final: $question');

    // Check if task request
    if (_isTaskRequest(question)) {
      _sendTaskRequest(question);
      return;
    }

    // Check simple goal-setting phrase
    final goal = _extractGoal(question);
    if (goal != null) {
      currentGoal.value = goal;
      _wsSend({'type': 'set_goal', 'goal': goal});
      return;
    }

    // Regular question
    _sendQuestionWithFrame(question);
  }

  bool _isTaskRequest(String text) {
    final q = text.toLowerCase();
    const triggers = [
      'take me to',
      'guide me to',
      'lead me to',
      'walk me to',
      'get me to',
      'navigate to',
      'i need to go',
      'i want to go',
      'i want to reach',
      'help me find',
      'find the',
      'find a',
      'i need to find',
      'i need to get',
      'go to',
      'bring me to',
    ];
    return triggers.any((t) => q.contains(t));
  }

  Future<void> _sendTaskRequest(String request) async {
    debugPrint('[Task] Initiating: $request');
    currentGoal.value = request; // show in UI immediately

    final frame = _latestFrame;
    if (frame == null) {
      _wsSend({'type': 'set_task', 'request': request});
      return;
    }

    try {
      final jpeg = await compute(
        _processYuvToJpeg,
        _YuvArgs.from(frame, _sensorOrientation),
      );
      _wsSend({
        'type': 'set_task',
        'request': request,
        if (jpeg != null) 'jpeg': base64Encode(jpeg),
      });
    } catch (_) {
      _wsSend({'type': 'set_task', 'request': request});
    }
  }

  Future<void> _sendQuestionWithFrame(String question) async {
    final frame = _latestFrame;
    if (frame == null) {
      _wsSend({'type': 'question', 'text': question});
      return;
    }

    try {
      final jpeg = await compute(
        _processYuvToJpeg,
        _YuvArgs.from(frame, _sensorOrientation),
      );
      _wsSend({
        'type': 'question',
        'text': question,
        if (jpeg != null) 'jpeg': base64Encode(jpeg),
      });
    } catch (_) {
      _wsSend({'type': 'question', 'text': question});
    }
  }

  String? _extractGoal(String question) {
    final q = question.toLowerCase();
    const triggers = [
      'take me to',
      'guide me to',
      'lead me to',
      'find the',
      'find a',
      'go to',
      'walk me to',
      'get me to',
      'navigate to',
      'i want to reach',
    ];
    for (final t in triggers) {
      if (q.contains(t)) {
        final idx = q.indexOf(t) + t.length;
        final goal = question
            .substring(idx)
            .trim()
            .replaceAll(RegExp(r'[?.!]$'), '');
        return goal.isNotEmpty ? goal : null;
      }
    }
    return null;
  }

  // ─── Toggle Assist (button tap) ───────────────────────────────────────────

  void toggleAssist() {
    switch (assistStatus.value) {
      case AssistStatus.idle:
        _startListening();
      case AssistStatus.stopped:
        assistStatus.value = AssistStatus.idle;
        _wsSend({'type': 'gesture', 'gesture': 'resume'});
      case AssistStatus.listening:
        _stt.stop();
        assistStatus.value = AssistStatus.idle;
      default:
        // Cancel everything
        _stt.stop();
        _audioPlayer.stop();
        _audioPlaying = false;
        assistStatus.value = AssistStatus.idle;
        _wsSend({'type': 'gesture', 'gesture': 'fist'});
    }
  }

  // ─── Worker Isolate ───────────────────────────────────────────────────────

  Future<void> _startWorkerIsolate() async {
    _workerReceivePort = ReceivePort();
    _workerIsolate = await Isolate.spawn(
      _yoloWorkerEntryPoint,
      _workerReceivePort!.sendPort,
    );
    final completer = Completer<void>();
    _workerReceivePort!.listen((msg) {
      if (msg is SendPort) {
        _workerSendPort = msg;
        completer.complete();
      } else if (msg is Uint8List?) {
        _yoloTaskCompleter?.complete(msg);
      }
    });
    return completer.future;
  }

  // ─── Camera ───────────────────────────────────────────────────────────────

  Future<void> _startCamera() async {
    final lensDir = isFrontCamera.value
        ? CameraLensDirection.front
        : CameraLensDirection.back;
    final desc = cameras.firstWhere(
      (c) => c.lensDirection == lensDir,
      orElse: () => cameras.first,
    );

    _camCtrl?.dispose();
    _camCtrl = CameraController(
      desc,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _camCtrl!.initialize();
      _sensorOrientation = desc.sensorOrientation;
      isCameraReady.value = true;
      await _camCtrl!.startImageStream(_onFrame);
    } catch (e) {
      loadingMessage.value = 'Camera error: $e';
    }
  }

  // ─── Frame pipeline ───────────────────────────────────────────────────────

  void _onFrame(CameraImage image) {
    _latestFrame = image;
    _frameCounter++;

    // Hand detection
    if (!_isProcessingHand &&
        isHandDetectionEnabled.value &&
        _handPlugin != null) {
      _isProcessingHand = true;
      try {
        final hands = _handPlugin!.detect(image, _sensorOrientation);
        HandGesture g = HandGesture.unknown;
        if (hands.isNotEmpty) {
          g = GestureDetector.detectGesture(hands.first);
        }
        handLandmarks.value = hands;

        final smoothed = _smoothGesture(g);
        currentGesture.value = smoothed;

        // Send gesture events to server
        _onGestureChanged(smoothed);
      } catch (e) {
        debugPrint('[Hand] $e');
      } finally {
        _isProcessingHand = false;
      }
    }

    // YOLO
    if (!_isProcessingYolo && _frameCounter % _yoloEveryN == 0) {
      _isProcessingYolo = true;
      _runYolo(image);
    }

    // FPS
    _fpsFrameCount++;
    final now = DateTime.now();
    final ms = now.difference(_lastFpsTime).inMilliseconds;
    if (ms >= 1000) {
      currentFps.value = _fpsFrameCount * 1000 / ms;
      _fpsFrameCount = 0;
      _lastFpsTime = now;
    }
  }

  Future<void> _runYolo(CameraImage image) async {
    if (_workerSendPort == null) {
      _isProcessingYolo = false;
      return;
    }
    try {
      _yoloTaskCompleter = Completer<Uint8List?>();
      _workerSendPort!.send(_YuvArgs.from(image, _sensorOrientation));
      final jpeg = await _yoloTaskCompleter!.future;
      if (jpeg == null) return;

      final result = await _yolo!.predict(
        jpeg,
        confidenceThreshold: confidenceThreshold.value,
        iouThreshold: iouThreshold.value,
      );

      final raw = result['detections'] as List? ?? [];
      final parsed = raw.map((d) {
        final res = YOLOResult.fromMap(Map<String, dynamic>.from(d));
        return BBox(
          label: res.className,
          confidence: res.confidence,
          left: res.normalizedBox.left,
          top: res.normalizedBox.top,
          right: res.normalizedBox.right,
          bottom: res.normalizedBox.bottom,
        );
      }).toList();

      yoloBoxes.value = parsed;
      detectionCount.value = parsed.length;
    } catch (e) {
      debugPrint('[YOLO] $e');
    } finally {
      _isProcessingYolo = false;
    }
  }

  // ─── Camera controls ──────────────────────────────────────────────────────

  Future<void> flipCamera() async {
    _frameSendTimer?.cancel();
    isCameraReady.value = false;
    isFrontCamera.value = !isFrontCamera.value;
    currentZoomLevel.value = 1.0;
    yoloBoxes.clear();
    handLandmarks.clear();
    currentGesture.value = HandGesture.unknown;
    _frameCounter = 0;
    await _startCamera();
    _startFrameTimer();
  }

  Future<void> setZoomLevel(double zoom) async {
    if ((currentZoomLevel.value - zoom).abs() < 0.01) return;
    currentZoomLevel.value = zoom;
    try {
      await _camCtrl?.setZoomLevel(zoom);
    } catch (_) {}
  }

  void toggleHandDetection() {
    isHandDetectionEnabled.value = !isHandDetectionEnabled.value;
    if (!isHandDetectionEnabled.value) {
      handLandmarks.clear();
      currentGesture.value = HandGesture.unknown;
      _gestureHistory.clear();
    }
  }

  void setConfidence(double v) {
    if ((confidenceThreshold.value - v).abs() > 0.005) {
      confidenceThreshold.value = v;
    }
  }

  void setIou(double v) {
    if ((iouThreshold.value - v).abs() > 0.005) iouThreshold.value = v;
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────

  HandGesture _smoothGesture(HandGesture g) {
    _gestureHistory.add(g);
    if (_gestureHistory.length > _historySize) _gestureHistory.removeAt(0);
    final counts = <HandGesture, int>{};
    for (final x in _gestureHistory) counts[x] = (counts[x] ?? 0) + 1;
    HandGesture best = HandGesture.unknown;
    int max = 0;
    counts.forEach((gesture, c) {
      if (c > max) {
        max = c;
        best = gesture;
      }
    });
    return best;
  }
}

// ─── Worker Isolate ───────────────────────────────────────────────────────────

void _yoloWorkerEntryPoint(SendPort mainSendPort) {
  final childPort = ReceivePort();
  mainSendPort.send(childPort.sendPort);
  childPort.listen((msg) {
    if (msg is _YuvArgs) mainSendPort.send(_processYuvToJpeg(msg));
  });
}

class _YuvArgs {
  final int width, height;
  final Uint8List yBytes, uBytes, vBytes;
  final int yRowStride, uvRowStride, uvPixelStride;
  final int sensorOrientation;

  const _YuvArgs({
    required this.width,
    required this.height,
    required this.yBytes,
    required this.uBytes,
    required this.vBytes,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.sensorOrientation,
  });

  factory _YuvArgs.from(CameraImage i, int o) => _YuvArgs(
    width: i.width,
    height: i.height,
    yBytes: i.planes[0].bytes,
    uBytes: i.planes.length > 1 ? i.planes[1].bytes : i.planes[0].bytes,
    vBytes: i.planes.length > 2
        ? i.planes[2].bytes
        : (i.planes.length > 1 ? i.planes[1].bytes : i.planes[0].bytes),
    yRowStride: i.planes[0].bytesPerRow,
    uvRowStride: i.planes.length > 1 ? i.planes[1].bytesPerRow : i.planes[0].bytesPerRow,
    uvPixelStride: i.planes.length > 1 ? (i.planes[1].bytesPerPixel ?? 1) : 1,
    sensorOrientation: o,
  );
}

Uint8List? _processYuvToJpeg(_YuvArgs a) {
  try {
    const int sw = 4;
    final int subW = a.width ~/ sw;
    final int subH = a.height ~/ sw;
    final bool isRot =
        (a.sensorOrientation == 90 || a.sensorOrientation == 270);
    final image = img.Image(
      width: isRot ? subH : subW,
      height: isRot ? subW : subH,
    );

    for (int sy = 0; sy < subH; sy++) {
      for (int sx = 0; sx < subW; sx++) {
        final x = sx * sw;
        final y = sy * sw;
        final yi = y * a.yRowStride + x;
        final yv = a.yBytes[yi];
        final uvx = x >> 1;
        final uvy = y >> 1;
        final uvi = uvy * a.uvRowStride + uvx * a.uvPixelStride;
        final u = a.uBytes[uvi] - 128;
        final v = a.vBytes[uvi] - 128;
        final r = (yv + ((114883 * v) >> 16)).clamp(0, 255);
        final g = (yv - ((28189 * u + 58509 * v) >> 16)).clamp(0, 255);
        final b = (yv + ((145161 * u) >> 16)).clamp(0, 255);

        int tx, ty;
        if (a.sensorOrientation == 90) {
          tx = subH - 1 - sy;
          ty = sx;
        } else if (a.sensorOrientation == 270) {
          tx = sy;
          ty = subW - 1 - sx;
        } else {
          tx = sx;
          ty = sy;
        }

        image.setPixelRgb(tx, ty, r, g, b);
      }
    }
    return Uint8List.fromList(img.encodeJpg(image, quality: 55));
  } catch (_) {
    return null;
  }
}
