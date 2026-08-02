// lib/app/modules/voice_assist/controller/voice_assist_controller.dart

import 'dart:async';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;

import '../../services/speech_service.dart';
import '../../services/tts_service.dart';

/// The states the voice assist module flows through.
enum VoiceAssistState {
  initializing, // Camera + services starting
  waitingForCapture, // "Show product and say Capture"
  listening, // Actively listening for "capture" command
  capturing, // Taking photo
  analyzingProduct, // Sending to backend for identification
  productIdentified, // Product identified, ready for question
  waitingForQuestion, // "What do you want to know?"
  listeningQuestion, // Listening for the user's question
  processingQuestion, // Sending question + image to backend
  playingResponse, // Playing ElevenLabs audio response
  complete, // Done, ready for another question or exit
  error, // Something went wrong
}

class VoiceAssistController extends GetxController {
  // ─── Configuration ──────────────────────────────────────────────────────
  static const String _baseUrl = String.fromEnvironment('SERVER_URL', defaultValue: 'http://10.0.2.2:8765');
  // static const String _baseUrl = 'http://localhost:8000'; // iOS simulator
  // static const String _baseUrl = 'http://YOUR_SERVER_IP:8000'; // Physical device

  // ─── Services ───────────────────────────────────────────────────────────
  late final TtsService _ttsService;
  late final SpeechService _speechService;

  // ─── Camera ─────────────────────────────────────────────────────────────
  CameraController? cameraController;
  final RxBool isCameraReady = false.obs;

  // ─── State ──────────────────────────────────────────────────────────────
  final Rx<VoiceAssistState> state = VoiceAssistState.initializing.obs;
  final RxString statusMessage = 'Initializing...'.obs;
  final RxString responseText = ''.obs;
  final RxString productName = ''.obs;
  final RxString userQuestion = ''.obs;
  final RxString partialSpeech = ''.obs;
  final RxBool isLoading = false.obs;
  final RxDouble loadingProgress = 0.0.obs;
  final RxInt processingTimeMs = 0.obs;

  // ─── Internal ───────────────────────────────────────────────────────────
  String? _capturedImageBase64;
  String? _sessionId;
  bool _disposed = false;

  // ─── Lifecycle ──────────────────────────────────────────────────────────

  @override
  void onInit() {
    super.onInit();
    _sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _initialize();
  }

  @override
  void onClose() {
    _disposed = true;
    cameraController?.dispose();
    super.onClose();
  }

  // ─── Initialization Flow ───────────────────────────────────────────────

  Future<void> _initialize() async {
    try {
      _setState(VoiceAssistState.initializing, 'Setting up camera...');

      // Initialize services
      _ttsService = Get.find<TtsService>();
      _speechService = Get.find<SpeechService>();

      // Initialize camera
      await _initCamera();

      // Small delay for camera to warm up
      await Future.delayed(const Duration(milliseconds: 500));

      // Start the flow
      await _startCaptureFlow();
    } catch (e) {
      _setState(
        VoiceAssistState.error,
        'Failed to initialize: ${e.toString()}',
      );
    }
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw Exception('No cameras available');
    }

    // Use back camera
    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    cameraController = CameraController(
      backCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await cameraController!.initialize();
    isCameraReady.value = true;
  }

  // ─── Main Flow: Capture ────────────────────────────────────────────────

  Future<void> _startCaptureFlow() async {
    if (_disposed) return;

    _setState(
      VoiceAssistState.waitingForCapture,
      'Show a product and say "Capture"',
    );

    // Speak the instruction
    await _ttsService.speak(
      'Show any product you need to know about in front of you, and say Capture.',
    );
    await _ttsService.waitUntilDone();

    if (_disposed) return;

    // Start listening for "capture" command
    await _listenForCaptureCommand();
  }

  Future<void> _listenForCaptureCommand() async {
    if (_disposed) return;

    _setState(VoiceAssistState.listening, 'Listening for "Capture"...');

    // Listen in a loop until we hear "capture"
    bool captureDetected = false;
    int attempts = 0;
    const maxAttempts = 5;

    while (!captureDetected && !_disposed && attempts < maxAttempts) {
      attempts++;

      final command = await _speechService.listenForCommand(
        timeout: const Duration(seconds: 8),
      );

      if (_disposed) return;

      final lowerCommand = command.toLowerCase().trim();

      if (lowerCommand.contains('capture') ||
          lowerCommand.contains('take') ||
          lowerCommand.contains('snap') ||
          lowerCommand.contains('photo') ||
          lowerCommand.contains('shoot') ||
          lowerCommand.contains('click')) {
        captureDetected = true;
      } else if (command.isNotEmpty) {
        // User said something but not "capture"
        await _ttsService.speak(
          'I didn\'t catch that. Say "Capture" when ready.',
        );
        await _ttsService.waitUntilDone();
      }
      // If empty (timeout), silently retry
    }

    if (_disposed) return;

    if (captureDetected) {
      await _captureImage();
    } else {
      // Max attempts reached
      await _ttsService.speak(
        'I couldn\'t hear the capture command. Tap the screen to capture instead.',
      );
      _setState(
        VoiceAssistState.waitingForCapture,
        'Tap screen to capture, or say "Capture"',
      );
    }
  }

  /// Manual capture via tap (accessibility fallback).
  Future<void> onScreenTap() async {
    if (state.value == VoiceAssistState.waitingForCapture ||
        state.value == VoiceAssistState.listening) {
      await _captureImage();
    } else if (state.value == VoiceAssistState.complete) {
      // Ask another question
      await _askForQuestion();
    }
  }

  Future<void> _captureImage() async {
    if (_disposed || cameraController == null || !isCameraReady.value) return;

    try {
      _setState(VoiceAssistState.capturing, 'Capturing...');

      // Haptic feedback
      HapticFeedback.heavyImpact();

      // Capture image
      final XFile imageFile = await cameraController!.takePicture();
      final Uint8List imageBytes = await imageFile.readAsBytes();
      _capturedImageBase64 = base64Encode(imageBytes);

      // Short confirmation
      await _ttsService.speak('Got it. Analyzing the product.');

      // Send to backend
      await _analyzeProduct();
    } catch (e) {
      _setState(VoiceAssistState.error, 'Capture failed: ${e.toString()}');
      await _ttsService.speak('Sorry, capture failed. Let\'s try again.');
      await Future.delayed(const Duration(seconds: 1));
      await _startCaptureFlow();
    }
  }

  // ─── Backend Communication ─────────────────────────────────────────────

  Future<void> _analyzeProduct() async {
    if (_disposed || _capturedImageBase64 == null) return;

    _setState(VoiceAssistState.analyzingProduct, 'Analyzing product...');
    isLoading.value = true;
    loadingProgress.value = 0.3;

    try {
      final stopwatch = Stopwatch()..start();

      final response = await http
          .post(
            Uri.parse('$_baseUrl/api/product/identify'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'image_base64': _capturedImageBase64,
              'question': '',
              'session_id': _sessionId,
            }),
          )
          .timeout(const Duration(seconds: 30));

      stopwatch.stop();
      processingTimeMs.value = stopwatch.elapsedMilliseconds;
      loadingProgress.value = 0.8;

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final text = data['text'] as String? ?? '';
        final audioB64 = data['audio_base64'] as String?;
        productName.value = data['product_name'] as String? ?? 'Product';

        responseText.value = text;
        _setState(VoiceAssistState.productIdentified, text);

        loadingProgress.value = 1.0;
        isLoading.value = false;

        // Play ElevenLabs audio if available, otherwise fallback to device TTS
        if (audioB64 != null && audioB64.isNotEmpty) {
          try {
            await _ttsService.playElevenLabsAudio(audioB64);
            await _ttsService.waitUntilDone();
          } catch (_) {
            await _ttsService.speak(text);
            await _ttsService.waitUntilDone();
          }
        } else {
          await _ttsService.speak(text);
          await _ttsService.waitUntilDone();
        }

        if (_disposed) return;

        // Now ask what they want to know
        await _askForQuestion();
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      isLoading.value = false;
      loadingProgress.value = 0.0;
      _setState(VoiceAssistState.error, 'Analysis failed: ${e.toString()}');
      await _ttsService.speak(
        'Sorry, I couldn\'t analyze the product. Let\'s try again.',
      );
      await Future.delayed(const Duration(seconds: 2));
      await _startCaptureFlow();
    }
  }

  // ─── Question Flow ─────────────────────────────────────────────────────

  Future<void> _askForQuestion() async {
    if (_disposed) return;

    _setState(VoiceAssistState.waitingForQuestion, 'What do you want to know?');

    await _ttsService.speak('What do you want to know about this product?');
    await _ttsService.waitUntilDone();

    if (_disposed) return;

    await _listenForQuestion();
  }

  Future<void> _listenForQuestion() async {
    if (_disposed) return;

    _setState(VoiceAssistState.listeningQuestion, 'Listening...');
    partialSpeech.value = '';

    final question = await _speechService.listenContinuous(
      timeout: const Duration(seconds: 15),
      onPartial: (partial) {
        partialSpeech.value = partial;
      },
    );

    if (_disposed) return;

    if (question.trim().isEmpty) {
      await _ttsService.speak(
        'I didn\'t hear your question. Please try again.',
      );
      await _ttsService.waitUntilDone();
      await _askForQuestion();
      return;
    }

    // Check for exit commands
    final lowerQ = question.toLowerCase().trim();
    if (lowerQ.contains('go back') ||
        lowerQ.contains('exit') ||
        lowerQ.contains('close') ||
        lowerQ.contains('done') ||
        lowerQ.contains('that\'s all') ||
        lowerQ.contains('no thanks')) {
      await _ttsService.speak('Okay, closing product assist.');
      await Future.delayed(const Duration(milliseconds: 500));
      Get.back();
      return;
    }

    // Check for new capture request
    if (lowerQ.contains('new product') ||
        lowerQ.contains('different product') ||
        lowerQ.contains('scan again') ||
        lowerQ.contains('another product')) {
      await _ttsService.speak('Okay, let\'s scan a new product.');
      await Future.delayed(const Duration(milliseconds: 500));
      await _startCaptureFlow();
      return;
    }

    userQuestion.value = question;
    await _processQuestion(question);
  }

  Future<void> _processQuestion(String question) async {
    if (_disposed || _capturedImageBase64 == null) return;

    _setState(VoiceAssistState.processingQuestion, 'Thinking...');
    isLoading.value = true;
    loadingProgress.value = 0.2;

    try {
      final stopwatch = Stopwatch()..start();

      // Simulate gradual progress
      _simulateProgress();

      final response = await http
          .post(
            Uri.parse('$_baseUrl/api/product/ask'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'image_base64': _capturedImageBase64,
              'question': question,
              'session_id': _sessionId,
            }),
          )
          .timeout(const Duration(seconds: 30));

      stopwatch.stop();
      processingTimeMs.value = stopwatch.elapsedMilliseconds;
      loadingProgress.value = 1.0;
      isLoading.value = false;

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final text = data['text'] as String? ?? '';
        final audioB64 = data['audio_base64'] as String?;

        responseText.value = text;
        _setState(VoiceAssistState.playingResponse, text);

        // Play response
        if (audioB64 != null && audioB64.isNotEmpty) {
          try {
            await _ttsService.playElevenLabsAudio(audioB64);
            await _ttsService.waitUntilDone();
          } catch (_) {
            await _ttsService.speak(text);
            await _ttsService.waitUntilDone();
          }
        } else {
          await _ttsService.speak(text);
          await _ttsService.waitUntilDone();
        }

        if (_disposed) return;

        // Ready for another question
        _setState(
          VoiceAssistState.complete,
          'Ask another question or say "done"',
        );

        await Future.delayed(const Duration(milliseconds: 800));
        if (!_disposed) {
          await _askForQuestion();
        }
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      isLoading.value = false;
      loadingProgress.value = 0.0;
      _setState(VoiceAssistState.error, 'Error: ${e.toString()}');
      await _ttsService.speak(
        'Sorry, I couldn\'t process that. Please ask again.',
      );
      await Future.delayed(const Duration(seconds: 1));
      await _askForQuestion();
    }
  }

  void _simulateProgress() {
    Timer.periodic(const Duration(milliseconds: 300), (timer) {
      if (!isLoading.value || _disposed) {
        timer.cancel();
        return;
      }
      if (loadingProgress.value < 0.85) {
        loadingProgress.value += 0.05;
      }
    });
  }

  // ─── Retry / Restart ───────────────────────────────────────────────────

  Future<void> restart() async {
    _capturedImageBase64 = null;
    responseText.value = '';
    productName.value = '';
    userQuestion.value = '';
    partialSpeech.value = '';
    isLoading.value = false;
    loadingProgress.value = 0.0;
    _sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    await _startCaptureFlow();
  }

  // ─── Helpers ───────────────────────────────────────────────────────────

  void _setState(VoiceAssistState newState, String message) {
    state.value = newState;
    statusMessage.value = message;
  }

  /// Whether we have a captured product image to show.
  bool get hasCapture => _capturedImageBase64 != null;

  /// Get the captured image bytes for display.
  Uint8List? get capturedImageBytes {
    if (_capturedImageBase64 == null) return null;
    return base64Decode(_capturedImageBase64!);
  }
}
