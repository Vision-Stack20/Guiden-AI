import 'dart:async';

import 'package:camera/camera.dart';
import 'package:get/get.dart';
import 'package:guiden/services/voice_assistant_controller.dart';
import 'package:sound_generator/sound_generator.dart';
import 'package:sound_generator/waveTypes.dart';

class LightFrequencyController extends GetxController {
  CameraController? cameraController;

  final isRunning = false.obs;
  final brightness = 0.0.obs;
  final frequency = 300.0.obs;

  double _smoothedBrightness = 0.0;
  static const double _smoothingFactor = 0.25;

  int _frameCount = 0;
  static const int _processEveryNthFrame = 2;

  late VoiceAssistantController _voiceAssistant;

  @override
  Future<void> onInit() async {
    super.onInit();

    // Get voice assistant
    _voiceAssistant = Get.find<VoiceAssistantController>();

    // High quality audio setup
    await SoundGenerator.init(48000);

    // SINE wave for clean, powerful bass
    SoundGenerator.setWaveType(waveTypes.SQUAREWAVE);

    // Higher volume for powerful sound
    SoundGenerator.setVolume(0.7);

    await _initCamera();

    // Wait for TTS to finish speaking the navigation message, then pause and start
    await Future.delayed(const Duration(milliseconds: 3000));
    _voiceAssistant.pauseVoiceAssistant();

    // Auto-start the frequency detector
    start();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final cam = cameras.firstWhere(
      (e) => e.lensDirection == CameraLensDirection.back,
    );

    cameraController = CameraController(
      cam,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await cameraController!.initialize();
    await cameraController!.setFocusMode(FocusMode.locked);
    await cameraController!.setExposureMode(ExposureMode.locked);
  }

  void start() {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      print("Camera not ready yet");
      return;
    }

    isRunning.value = true;

    // Start the tone
    SoundGenerator.play();

    final initialFreq = _mapBrightnessToFrequency(_smoothedBrightness);
    SoundGenerator.setFrequency(initialFreq);
    frequency.value = initialFreq;

    cameraController!.startImageStream((image) {
      _frameCount++;

      if (_frameCount % _processEveryNthFrame != 0) return;

      final b = _calculateBrightnessFast(image);

      // Smooth transitions
      _smoothedBrightness =
          (_smoothingFactor * b) +
          ((1 - _smoothingFactor) * _smoothedBrightness);

      brightness.value = _smoothedBrightness;

      // Professional frequency mapping
      final f = _mapBrightnessToFrequency(_smoothedBrightness);
      frequency.value = f;

      SoundGenerator.setFrequency(f);
    });
  }

  void stop() async {
    isRunning.value = false;
    _frameCount = 0;
    _smoothedBrightness = 0.0;

    SoundGenerator.stop();
    await cameraController?.stopImageStream();
  }

  double _calculateBrightnessFast(CameraImage image) {
    final plane = image.planes[0];
    final bytes = plane.bytes;
    final stride = plane.bytesPerRow;

    int sum = 0;
    int count = 0;

    for (int row = 0; row < image.height; row += 5) {
      final rowStart = row * stride;
      for (int col = 0; col < image.width; col += 20) {
        if (rowStart + col < bytes.length) {
          sum += bytes[rowStart + col];
          count++;
        }
      }
    }

    return count > 0 ? (sum / count) / 255.0 : 0.0;
  }

  // SPEAKER CLEANING STYLE - Deep bass to high frequency sweep
  double _mapBrightnessToFrequency(double b) {
    // Dark → Deep bass (20Hz - feel the vibration)
    // Bright → High frequency (800Hz - clear tone)

    const double minFreq = 20.0; // Deep bass - speaker cleaning range
    const double maxFreq = 800.0; // High clear tone

    // Linear mapping for consistent sweep (like speaker cleaners)
    return minFreq + (b * (maxFreq - minFreq));
  }

  @override
  void onClose() {
    stop();
    SoundGenerator.release();
    cameraController?.dispose();

    // Resume voice assistant and announce
    _voiceAssistant.resumeVoiceAssistant();
    Future.delayed(const Duration(milliseconds: 500), () {
      _voiceAssistant.speak(
        "Back to home screen. You can use navigate to go to gesture test or YOLO detection.",
      );
    });

    super.onClose();
  }
}
