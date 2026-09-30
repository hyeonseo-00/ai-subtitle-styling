// lib/services/api_service.dart

import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/caption_mode.dart';

class ApiService {
  // [중요] Flutter Web은 브라우저와 같으므로 localhost 사용 가능
  static const String baseUrl = 'http://127.0.0.1:8000';

  // SUB-01: 자막 모드 리스트 조회
  Future<List<CaptionMode>> fetchModes(int profileId) async {
    final url = Uri.parse('$baseUrl/subtitles/profiles/$profileId/modes');

    try {
      final response = await http.get(url);

      if (response.statusCode == 200) {
        // UTF-8 디코딩 처리 (한글 깨짐 방지)
        List<dynamic> body = jsonDecode(utf8.decode(response.bodyBytes));
        return body.map((dynamic item) => CaptionMode.fromJson(item)).toList();
      } else {
        throw Exception('Failed to load modes: ${response.statusCode}');
      }
    } catch (e) {
      print("API Error: $e");
      throw Exception('Connection Failed. Is backend running?');
    }
  }

  // QI-01: 자막 모드 적용
  Future<bool> applyMode(int profileId, int modeId) async {
    final url = Uri.parse(
      '$baseUrl/subtitles/profiles/$profileId/apply-mode/$modeId',
    );

    try {
      final response = await http.post(url);
      return response.statusCode == 200;
    } catch (e) {
      print("Apply Error: $e");
      return false;
    }
  }
}
