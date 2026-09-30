# 중간점검 성공코드
 
import pyaudio
import numpy as np
import threading
import queue
import tkinter as tk
from faster_whisper import WhisperModel
import torch
import time
 
# --- 설정 ---
MODEL_SIZE = "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
 
# 오디오 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
 
# 큐 (스레드 간 데이터 전달용)
audio_queue = queue.Queue()
text_queue = queue.Queue()
word_queue = queue.Queue()  # 단어별 정보 큐

# 공유 변수 (현재 목소리 크기)
current_volume = 0.0
current_sentence = []  # 현재 문장의 단어들
sentence_complete_time = 0  # 문장 완료 시간 (0이면 미완료)
 
class SubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FCB Chicago Style Subtitles")
        
        # 전체 화면 크기 가져오기
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # 창을 화면 하단 중앙에 배치 (너비는 화면의 80%, 높이는 150px)
        window_width = int(screen_width * 0.8)
        window_height = 150
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50  # 하단에서 50px 위
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#000000')
        
        # 항상 위에 띄우기 & 반투명 배경
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.85)  # 약간 투명한 배경
        
        # 창 테두리 제거 (선택사항)
        # self.root.overrideredirect(True)
        
        # 메인 컨테이너 프레임 (중앙 정렬)
        main_frame = tk.Frame(root, bg='#000000')
        main_frame.pack(expand=True, fill='both')
        
        # 단어별 Label을 담을 프레임 (중앙 정렬, 가로 배치)
        self.word_frame = tk.Frame(main_frame, bg='#000000')
        self.word_frame.pack(expand=True)
        
        # 단어 Label들을 저장할 리스트
        self.word_labels = []
        self.word_animations = {}  # 단어별 애니메이션 정보

        # GUI 업데이트 루프 시작
        self.update_ui()
 
    def update_ui(self):
        """큐에서 단어 정보를 가져와 화면 갱신"""
        global sentence_complete_time
        
        # 문장 완료 후 1.5초 후 화면 지우기
        if sentence_complete_time > 0:
            if time.time() - sentence_complete_time > 1.5:
                self.clear_sentence()
                sentence_complete_time = 0
        
        try:
            # 단어 큐 확인 (새로운 단어가 있으면 추가)
            while not word_queue.empty():
                word_info = word_queue.get_nowait()
                word, volume = word_info
                self.add_word_with_animation(word, volume)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Update error: {e}")

        # 애니메이션 업데이트
        self.update_animations()
       
        # 0.03초마다 반복 (더 빠른 업데이트)
        self.root.after(30, self.update_ui)
    
    def clear_sentence(self):
        """현재 문장의 모든 단어 Label 제거"""
        for label in self.word_labels:
            label.destroy()
        self.word_labels = []
        self.word_animations = {}
    
    def add_word_with_animation(self, word, volume):
        """단어를 애니메이션과 함께 추가 (Caption with Intention 스타일)"""
        # 의도(intention)에 따른 기본 스타일 결정
        intention_level = self.get_intention_level(volume)
        
        # 감정 분석
        emotion = self.detect_emotion(word)
        
        # 볼륨에 따라 크기 결정 (볼륨이 클수록 더 크게, 고정 크기)
        # 볼륨을 직접 사용하여 크기 계산
        if volume < 100:
            font_size = 22  # 조용한 목소리
        elif volume < 300:
            font_size = 26  # 보통 목소리
        elif volume < 600:
            font_size = 32  # 큰 목소리
        else:
            font_size = 40  # 매우 큰 목소리
        
        # 단어 Label 생성 - 볼륨에 따라 결정된 크기로 고정
        label = tk.Label(self.word_frame, text=word + " ", 
                        font=("Segoe UI", font_size, "normal"),
                        bg='#000000', fg='#FFFFFF')
        label.pack(side=tk.LEFT, padx=3)
        self.word_labels.append(label)
        
        # 애니메이션 정보 저장 (색상만 애니메이션, 크기는 고정)
        word_id = len(self.word_labels) - 1
        self.word_animations[word_id] = {
            'animation_step': 0,
            'max_steps': 20,  # 애니메이션 단계 수
            'intention_level': intention_level,  # 의도 레벨 저장
            'font_size': font_size,  # 고정된 폰트 크기
            'volume': volume,
            'emotion': emotion  # 감정 정보 저장
        }
    
    def detect_emotion(self, word):
        """단어에서 감정 감지"""
        # 부정적 감정 키워드
        negative_keywords = ['짜증', '화', '화나', '싫', '미워', '답답', '스트레스', 
                            '힘들', '아픈', '슬프', '우울', '불안', '걱정', '두려',
                            '싫어', '미워', '증오', '분노', '열받', '빡쳐', '짜증나',
                            '짜증나', '짜증내', '화나', '화내', '빡치', '열받아']
        
        # 긍정적 감정 키워드
        positive_keywords = ['좋아', '행복', '기쁘', '사랑', '만족', '즐거', '신나',
                            '기쁘', '행복', '좋아', '사랑', '즐거', '신나', '만족',
                            '기쁘', '행복해', '좋아해', '사랑해', '즐거워', '신나']
        
        word_lower = word.lower()
        
        # 부정적 감정 체크
        for keyword in negative_keywords:
            if keyword in word:
                return 'negative'
        
        # 긍정적 감정 체크
        for keyword in positive_keywords:
            if keyword in word:
                return 'positive'
        
        return 'neutral'  # 중립
    
    def get_intention_level(self, volume):
        """볼륨에 따라 의도 레벨 반환 (0-3)"""
        if volume < 100:
            return 0  # 낮은 의도
        elif volume < 300:
            return 1  # 보통 의도
        elif volume < 600:
            return 2  # 높은 의도
        else:
            return 3  # 매우 높은 의도
    
    def update_animations(self):
        """모든 단어의 애니메이션 업데이트 (색상만 애니메이션, 크기는 고정)"""
        for word_id, anim_info in list(self.word_animations.items()):
            if word_id < len(self.word_labels):
                label = self.word_labels[word_id]
                step = anim_info['animation_step']
                max_steps = anim_info['max_steps']
                intention = anim_info['intention_level']
                font_size = anim_info['font_size']  # 고정된 폰트 크기
                emotion = anim_info['emotion']
                
                # 색상만 애니메이션 (크기는 변경하지 않음)
                if step < max_steps:
                    progress = step / max_steps
                    
                    # 감정이 우선: 부정적 감정은 빨간색
                    if emotion == 'negative':
                        # 부정적 감정: 빨간색 계열
                        if progress < 0.3:
                            r, g, b = 255, 100, 100  # 밝은 빨간색
                        elif progress < 0.7:
                            t = (progress - 0.3) / 0.4
                            r = int(255 - (255 - 255) * t)
                            g = int(100 + (150 - 100) * t)
                            b = int(100 + (150 - 100) * t)
                        else:
                            r, g, b = 255, 150, 150  # 부드러운 빨간색 유지
                    elif emotion == 'positive':
                        # 긍정적 감정: 초록색/파란색 계열
                        if progress < 0.3:
                            r, g, b = 150, 255, 200  # 밝은 초록색
                        elif progress < 0.7:
                            t = (progress - 0.3) / 0.4
                            r = int(150 + (200 - 150) * t)
                            g = int(255 - (255 - 255) * t)
                            b = int(200 + (255 - 200) * t)
                        else:
                            r, g, b = 200, 255, 255  # 밝은 청록색
                    else:
                        # 중립 감정: 의도에 따른 색상
                        if intention == 0:  # 낮은 의도 - 부드러운 흰색
                            if progress < 0.4:
                                r, g, b = 255, 255, 240  # 매우 밝은 흰색
                            else:
                                r, g, b = 255, 255, 255  # 순수 흰색
                        elif intention == 1:  # 보통 의도 - 약간 노란색
                            if progress < 0.3:
                                r, g, b = 255, 255, 220  # 밝은 노란색
                            elif progress < 0.7:
                                t = (progress - 0.3) / 0.4
                                r, g, b = 255, 255, int(220 + (255 - 220) * t)
                            else:
                                r, g, b = 255, 255, 255
                        elif intention == 2:  # 높은 의도 - 밝은 노란색
                            if progress < 0.3:
                                r, g, b = 255, 255, 180  # 밝은 노란색
                            elif progress < 0.7:
                                t = (progress - 0.3) / 0.4
                                r, g, b = 255, 255, int(180 + (255 - 180) * t)
                            else:
                                r, g, b = 255, 255, 255
                        else:  # 매우 높은 의도 - 강렬한 노란색
                            if progress < 0.3:
                                r, g, b = 255, 255, 150  # 강렬한 노란색
                            elif progress < 0.7:
                                t = (progress - 0.3) / 0.4
                                r, g, b = 255, 255, int(150 + (255 - 150) * t)
                            else:
                                r, g, b = 255, 255, 255
                    
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    # 색상만 변경, 크기는 고정
                    label.config(fg=color, font=("Segoe UI", font_size, "normal"))
                    anim_info['animation_step'] += 1
                else:
                    # 애니메이션 완료 - 감정과 의도에 따른 최종 색상 (크기는 그대로)
                    if emotion == 'negative':
                        final_color = '#FF6666'  # 부정적 감정: 빨간색 유지
                    elif emotion == 'positive':
                        final_color = '#99FFCC'  # 긍정적 감정: 초록색 유지
                    else:
                        final_color = '#FFFFFF'
                        if intention >= 2:
                            final_color = '#FFFFE0'  # 높은 의도는 약간 노란색 유지
                    
                    label.config(fg=final_color, font=("Segoe UI", font_size, "normal"))
    
 
 
def calculate_rms(audio_data):
    """오디오 데이터의 에너지(RMS) 계산"""
    # int16 -> float 변환
    data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)
    rms = np.sqrt(np.mean(data**2))
    return rms
 
def audio_thread_func():
    """마이크 입력 및 볼륨 측정 스레드"""
    global current_volume
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
   
    while True:
        data = stream.read(CHUNK)
        audio_queue.put(data)
       
        # 실시간 볼륨 측정 (GUI용)
        rms = calculate_rms(data)
        # 부드러운 전환을 위해 약간의 보정
        current_volume = (current_volume * 0.6) + (rms * 0.4)
 
def whisper_thread_func():
    """STT 추론 스레드"""
    global current_sentence, sentence_complete_time, current_volume
    
    print(f"[*] 모델 로딩 중... ({DEVICE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("[OK] 모델 준비 완료")

    accumulated_audio = np.array([], dtype=np.float32)
    last_inference_time = time.time()
    last_text = ""  # 이전 텍스트 추적
    audio_buffer_for_volume = []  # 볼륨 계산용 오디오 버퍼
   
    while True:
        # 오디오 데이터 누적
        chunk_volume = 0.0
        while not audio_queue.empty():
            data = audio_queue.get()
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            accumulated_audio = np.concatenate((accumulated_audio, audio_array))
            
            # 볼륨 계산용 버퍼
            audio_buffer_for_volume.append(data)
            if len(audio_buffer_for_volume) > 10:  # 최근 10개 청크만 유지
                audio_buffer_for_volume.pop(0)
            
            # 현재 청크의 볼륨 계산
            rms = calculate_rms(data)
            chunk_volume = max(chunk_volume, rms)

        # 0.3초마다 추론 (더 빠른 반응)
        if time.time() - last_inference_time > 0.3:
            last_inference_time = time.time()
           
            if len(accumulated_audio) > RATE * 0.3:  # 0.3초 이상 데이터가 있을 때
                segments, _ = model.transcribe(accumulated_audio, language="ko", beam_size=1, 
                                              vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
               
                current_text = ""
                for segment in segments:
                    current_text += segment.text
                
                # 새로운 텍스트가 있으면 처리
                if current_text.strip() and current_text != last_text:
                    # 단어별로 분리
                    words = current_text.split()
                    
                    # 이전 텍스트와 비교하여 새로 추가된 단어만 처리
                    last_words = last_text.split() if last_text else []
                    new_words = words[len(last_words):] if len(words) > len(last_words) else words
                    
                        # 각 단어를 큐에 추가 (현재 볼륨과 함께)
                    # 단어별로 약간 다른 볼륨 적용 (더 자연스러운 효과)
                    for i, word in enumerate(new_words):
                        # 단어 순서에 따라 볼륨 약간 조정
                        word_volume = chunk_volume * (1.0 + (i % 3) * 0.1)
                        word_queue.put((word, word_volume))
                        current_sentence.append((word, word_volume))
                    
                    # 문장 종료 감지 (마침표, 느낌표, 물음표)
                    if any(punct in current_text for punct in ['.', '!', '?', '。', '！', '？']):
                        # 문장 완료 시간 기록 (1.5초 후 지우기)
                        sentence_complete_time = time.time()
                        current_sentence = []  # 현재 문장 초기화
                        # 버퍼 초기화하여 다음 문장 시작
                        accumulated_audio = np.array([], dtype=np.float32)
                        last_text = ""  # 다음 문장을 위해 초기화
                    
                    last_text = current_text

            # 버퍼 관리 (너무 길어지면 초기화 - 5초 기준)
            if len(accumulated_audio) > RATE * 5:
                accumulated_audio = np.array([], dtype=np.float32)
                last_text = ""
 
def main():
    # GUI 루트 생성
    root = tk.Tk()
    app = SubtitleApp(root)
 
    # 스레드 시작
    t1 = threading.Thread(target=audio_thread_func, daemon=True)
    t2 = threading.Thread(target=whisper_thread_func, daemon=True)
    t1.start()
    t2.start()
 
    # GUI 실행
    root.mainloop()
 
if __name__ == "__main__":
    main()