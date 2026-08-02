// lib/app/data/services/tts_service.dart

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_tts/flutter_tts.dart';
import 'package:get/get.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';

/// Handles both local TTS (flutter_tts) and ElevenLabs audio playback.
class TtsService extends GetxService {
  final FlutterTts _flutterTts = FlutterTts();
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription? _playerSub;

  final RxBool isSpeaking = false.obs;

  Future<TtsService> init() async {
    await _flutterTts.setLanguage('en-US');
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setPitch(1.0);
    await _flutterTts.setVolume(1.0);

    _flutterTts.setStartHandler(() => isSpeaking.value = true);
    _flutterTts.setCompletionHandler(() => isSpeaking.value = false);
    _flutterTts.setCancelHandler(() => isSpeaking.value = false);
    _flutterTts.setErrorHandler((msg) {
      isSpeaking.value = false;
    });

    return this;
  }

  /// Speak using device TTS (for quick prompts).
  Future<void> speak(String text) async {
    if (text.trim().isEmpty) return;
    await stop();
    isSpeaking.value = true;
    await _flutterTts.speak(text);
  }

  /// Play ElevenLabs audio from base64 MP3 data.
  Future<void> playElevenLabsAudio(String base64Audio) async {
    try {
      await stop();
      isSpeaking.value = true;

      final Uint8List audioBytes = base64Decode(base64Audio);
      final tempDir = await getTemporaryDirectory();
      final filePath =
          '${tempDir.path}/elevenlabs_${DateTime.now().millisecondsSinceEpoch}.mp3';
      final file = File(filePath);
      await file.writeAsBytes(audioBytes);

      await _audioPlayer.setFilePath(filePath);
      _playerSub?.cancel();
      _playerSub = _audioPlayer.playerStateStream.listen((state) {
        if (state.processingState == ProcessingState.completed) {
          isSpeaking.value = false;
          // Clean up temp file
          file.delete().catchError((_) {});
        }
      });

      await _audioPlayer.play();
    } catch (e) {
      isSpeaking.value = false;
      // Fallback to device TTS is handled by caller
      rethrow;
    }
  }

  /// Stop all audio.
  Future<void> stop() async {
    _playerSub?.cancel();
    _playerSub = null;
    await _flutterTts.stop();
    await _audioPlayer.stop();
    isSpeaking.value = false;
  }

  /// Wait until current speech/audio finishes.
  Future<void> waitUntilDone() async {
    while (isSpeaking.value) {
      await Future.delayed(const Duration(milliseconds: 100));
    }
  }

  @override
  void onClose() {
    _playerSub?.cancel();
    _flutterTts.stop();
    _audioPlayer.dispose();
    super.onClose();
  }
}
