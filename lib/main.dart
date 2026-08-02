import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_refresh_rate_control/flutter_refresh_rate_control.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:get/get.dart';
import 'package:guiden/routes/app_pages.dart';
import 'package:guiden/services/global_audio_controller.dart';
import 'package:guiden/services/voice_assistant_controller.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  // Initialize audio controller
  await Get.putAsync(() => AudioController().init());

  // Initialize voice assistant but don't start it yet (will start after splash)
  Get.put(VoiceAssistantController(), permanent: true);

  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Failed to get available cameras: $e');
    cameras = <CameraDescription>[];
  }
  final _refreshRateControl = FlutterRefreshRateControl();

  try {
    bool success = await _refreshRateControl.requestHighRefreshRate();
    if (success) {
      print('High refresh rate enabled');
    } else {
      print('Failed to enable high refresh rate');
    }
  } catch (e) {
    print('Error: $e');
  }

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ScreenUtilInit(
      designSize: const Size(360, 690),
      minTextAdapt: true,
      splitScreenMode: true,
      builder: (_, child) {
        return GetMaterialApp(
          debugShowCheckedModeBanner: false,
          title: 'Guiden App',
          initialRoute: AppPages.INITIAL,
          getPages: AppPages.routes,
        );
      },
    );
  }
}
