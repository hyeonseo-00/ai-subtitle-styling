// lib/main.dart

import 'package:flutter/material.dart';
import 'models/caption_mode.dart';
import 'services/api_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'The One Point TV',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212), // TV 배경색
        primaryColor: const Color(0xFFA50034), // LG Red
      ),
      home: const TvHomeScreen(),
    );
  }
}

class TvHomeScreen extends StatefulWidget {
  const TvHomeScreen({super.key});

  @override
  State<TvHomeScreen> createState() => _TvHomeScreenState();
}

class _TvHomeScreenState extends State<TvHomeScreen> {
  final ApiService _apiService = ApiService();
  late Future<List<CaptionMode>> _modesFuture;

  // 테스트용 프로필 ID (init_db.py로 생성된 데이터)
  final int _profileId = 1;
  int? _activeModeId;

  @override
  void initState() {
    super.initState();
    _refreshModes();
  }

  void _refreshModes() {
    setState(() {
      _modesFuture = _apiService.fetchModes(_profileId);
    });
  }

  Future<void> _handleModeSelect(CaptionMode mode) async {
    bool success = await _apiService.applyMode(_profileId, mode.id);
    if (success) {
      setState(() {
        _activeModeId = mode.id;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("'${mode.modeName}' 모드가 적용되었습니다."),
          backgroundColor: const Color(0xFFA50034),
          duration: const Duration(seconds: 1),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('자막 스타일 설정 (DX Project)'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: Center(
        child: FutureBuilder<List<CaptionMode>>(
          future: _modesFuture,
          builder: (context, snapshot) {
            if (snapshot.hasData) {
              final modes = snapshot.data!;
              return _buildModeGrid(modes);
            } else if (snapshot.hasError) {
              return Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, color: Colors.red, size: 60),
                  const SizedBox(height: 20),
                  Text(
                    '백엔드 연결 실패!\n${snapshot.error}',
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: _refreshModes,
                    child: const Text("재시도"),
                  ),
                ],
              );
            }
            return const CircularProgressIndicator();
          },
        ),
      ),
    );
  }

  Widget _buildModeGrid(List<CaptionMode> modes) {
    // 넓은 화면용 GridView
    return Container(
      constraints: const BoxConstraints(maxWidth: 1000),
      padding: const EdgeInsets.all(32),
      child: GridView.builder(
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3, // 한 줄에 3개씩
          childAspectRatio: 1.5, // 가로로 긴 카드
          crossAxisSpacing: 24,
          mainAxisSpacing: 24,
        ),
        itemCount: modes.length,
        itemBuilder: (context, index) {
          final mode = modes[index];
          final isActive = _activeModeId == mode.id;

          return InkWell(
            onTap: () => _handleModeSelect(mode),
            borderRadius: BorderRadius.circular(16),
            child: Container(
              decoration: BoxDecoration(
                color: isActive
                    ? const Color(0xFF2C2C2C)
                    : const Color(0xFF1E1E1E),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: isActive
                      ? const Color(0xFFA50034)
                      : Colors.transparent,
                  width: 3,
                ),
                boxShadow: isActive
                    ? [
                        BoxShadow(
                          color: const Color(0xFFA50034).withOpacity(0.4),
                          blurRadius: 15,
                        ),
                      ]
                    : [],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    mode.modeName,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    alignment: WrapAlignment.center,
                    children: [
                      if (mode.isEmpathyOn) _buildTag("감성 ON", Colors.amber),
                      if (mode.speaker) _buildTag("화자분리", Colors.blueAccent),
                      _buildTag("${mode.fontSize}px", Colors.grey),
                    ],
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildTag(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(text, style: TextStyle(color: color, fontSize: 12)),
    );
  }
}
