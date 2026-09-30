class CaptionMode {
  final int id;
  final String modeName;
  final bool isEmpathyOn;
  final int fontSize;
  final bool fontSizeToggle;
  final String fontColor;
  final bool fontColorToggle;
  final int fontLevel;
  final int colorLevel;
  final bool speaker;
  final bool bgm;
  final bool effect;

  CaptionMode({
    required this.id,
    required this.modeName,
    required this.isEmpathyOn,
    required this.fontSize,
    required this.fontSizeToggle,
    required this.fontColor,
    required this.fontColorToggle,
    required this.fontLevel,
    required this.colorLevel,
    required this.speaker,
    required this.bgm,
    required this.effect,
  });

  factory CaptionMode.fromJson(Map<String, dynamic> json) {
    return CaptionMode(
      id: json['id'],
      modeName: json['mode_name'],
      isEmpathyOn: json['is_empathy_on'] ?? false,
      fontSize: json['font_size'] ?? 20,
      fontSizeToggle: json['fontSize_toggle'] ?? false,
      fontColor: json['font_color'] ?? "#FFFFFF",
      fontColorToggle: json['fontColor_toggle'] ?? false,
      fontLevel: json['font_level'] ?? 1,
      colorLevel: json['color_level'] ?? 1,
      speaker: json['speaker'] ?? false,
      bgm: json['bgm'] ?? false,
      effect: json['effect'] ?? false,
    );
  }
}
