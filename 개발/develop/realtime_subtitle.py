import pyaudio
import numpy as np
import threading
import queue
import tkinter as tk
from faster_whisper import WhisperModel
import torch
import time

# ========================
# 0. 기본 설정 & 프리셋
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모드 선택: "fast", "balanced", "quality"
MODE_PRESET = "balanced"

if MODE_PRESET == "fast":
    MODEL_SIZE = "tiny"
    COMPUTE_TYPE = "int8" if DEVICE == "cuda" else "int8"
    INFER_INTERVAL = 0.1       # 추론 간격(초) - LG TV처럼 빠르게
    MIN_AUDIO_SEC = 0.0        # 최소 오디오 길이 제거 (말하는 동안 계속 추론)
    MAX_AUDIO_SEC = 2.0        # 버퍼 최대 길이(초)
elif MODE_PRESET == "balanced":
    MODEL_SIZE = "base"
    COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"
    INFER_INTERVAL = 0.15      # 추론 간격(초) - LG TV처럼 빠르게
    MIN_AUDIO_SEC = 0.0        # 최소 오디오 길이 제거
    MAX_AUDIO_SEC = 3.0
else:  # "quality"
    MODEL_SIZE = "small"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    INFER_INTERVAL = 0.2       # 추론 간격(초) - LG TV처럼 빠르게
    MIN_AUDIO_SEC = 0.0        # 최소 오디오 길이 제거
    MAX_AUDIO_SEC = 4.0

# ========================
# 1. 오디오 설정
# ========================
CHUNK = 512              # 너무 작으면 오류, 512~1024 사이 추천
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# ========================
# 2. 큐 & 공유 변수
# ========================
audio_queue = queue.Queue()
text_queue = queue.Queue()

current_volume = 0.0


# ========================
# 3. GUI (자막 표시용)
# ========================
class SubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Subtitles")

        # 화면 하단 가운데 작은 바 형태
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        window_width = int(screen_width * 0.8)
        window_height = 140
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 40

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg="#000000")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)

        main_frame = tk.Frame(root, bg="#000000")
        main_frame.pack(expand=True, fill="both")

        # 자막 한 줄 표시용 라벨
        self.subtitle_label = tk.Label(
            main_frame,
            text="듣고 있습니다...",
            font=("Malgun Gothic", 28, "normal"),
            bg="#000000",
            fg="#FFFFFF",
            wraplength=window_width - 40,
            justify="center"
        )
        self.subtitle_label.pack(expand=True, fill="both", padx=20, pady=20)

        self.update_ui()

    def update_ui(self):
        """텍스트 큐에서 최신 자막을 꺼내 화면에 표시"""
        try:
            # 큐에 여러 개가 쌓여 있으면, 마지막 것만 사용
            last_text = None
            queue_count = 0
            while not text_queue.empty():
                last_text = text_queue.get_nowait()
                queue_count += 1

            if last_text is not None:
                # tkinter Label에 실제로 텍스트 설정
                self.subtitle_label.config(text=last_text)
                # 업데이트 강제 실행
                self.root.update_idletasks()
                print(f"[UI 업데이트] 큐에서 {queue_count}개 가져옴, 표시: '{last_text}'")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[UI] error: {e}")
            import traceback
            traceback.print_exc()

        # 30ms마다 업데이트 (약 33fps)
        self.root.after(30, self.update_ui)


# ========================
# 4. 유틸: RMS 계산
# ========================
def calculate_rms(audio_data):
    """오디오 데이터의 에너지(RMS) 계산"""
    data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)
    rms = np.sqrt(np.mean(data ** 2))
    return rms


# ========================
# 5. 오디오 입력 스레드
# ========================
def audio_thread_func():
    """마이크 입력 + 볼륨 측정 스레드"""
    global current_volume

    try:
        p = pyaudio.PyAudio()
        
        # 사용 가능한 마이크 디바이스 확인
        print(f"[오디오] 사용 가능한 입력 디바이스:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("[오디오] 마이크 입력 시작됨")
        
        chunk_count = 0
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
            chunk_count += 1

            # 볼륨 계산 (필요하면 나중에 스타일링에 사용)
            rms = calculate_rms(data)
            current_volume = (current_volume * 0.6) + (rms * 0.4)
            
            # 100개 청크마다 볼륨 출력 (약 3초마다)
            if chunk_count % 100 == 0:
                print(f"[오디오] 볼륨: {current_volume:.1f}, 큐 크기: {audio_queue.qsize()}")
    except Exception as e:
        print(f"[오디오] 오류: {e}")
        import traceback
        traceback.print_exc()


# ========================
# 6. Whisper STT 스레드
# ========================
def whisper_thread_func():
    """실시간 STT 추론 스레드 (한 줄 자막만 출력)"""
    print(f"[*] Whisper 모델 로딩 중... ({DEVICE}, {MODEL_SIZE}, {COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("[OK] 모델 준비 완료")

    accumulated_audio = np.array([], dtype=np.float32)
    last_inference_time = time.time()
    last_text = ""

    while True:
        # 오디오 버퍼에서 가능한 만큼 가져오기 (ver2.py처럼)
        while not audio_queue.empty():
            data = audio_queue.get()  # ver2.py처럼 get() 사용
            audio_array = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            accumulated_audio = np.concatenate((accumulated_audio, audio_array))

        # 버퍼 관리 (ver2.py처럼 5초 기준)
        if len(accumulated_audio) > RATE * 5:
            accumulated_audio = np.array([], dtype=np.float32)
            last_text = ""

        # 0.3초마다 추론 (ver2.py와 동일)
        if time.time() - last_inference_time > 0.3:
            last_inference_time = time.time()

            # 최소 길이 체크 (ver2.py처럼 0.3초 이상 필요)
            if len(accumulated_audio) > RATE * 0.3:
                try:
                    # ver2.py와 동일한 설정으로 추론
                    segments, info = model.transcribe(
                        accumulated_audio,
                        language="ko",
                        beam_size=1,
                        vad_filter=True,  # ver2.py처럼 VAD 필터 사용
                        vad_parameters=dict(min_silence_duration_ms=500),  # ver2.py처럼 500ms
                    )

                    current_text = ""
                    segment_count = 0
                    for seg in segments:
                        current_text += seg.text
                        segment_count += 1

                    current_text = current_text.strip()
                    
                    # 디버깅: segments 정보 출력
                    if segment_count == 0:
                        print(f"[Whisper] segments가 비어있음 (오디오 길이: {len(accumulated_audio)/RATE:.2f}초)")
                    elif not current_text:
                        print(f"[Whisper] segments는 {segment_count}개 있지만 텍스트가 비어있음")

                    # 새 텍스트가 이전과 다르면 큐에 넣어서 UI 갱신
                    if current_text and current_text != last_text:
                        text_queue.put(current_text)
                        print(f"[인식] {current_text}")  # 디버깅 로그
                        last_text = current_text
                    elif current_text == last_text and current_text:
                        # 같은 텍스트지만 큐에 넣어서 UI 갱신 (확실하게 표시)
                        text_queue.put(current_text)

                except Exception as e:
                    print(f"[Whisper] error: {e}")
                    # 에러 나면 한 번 비워주고 계속
                    accumulated_audio = np.array([], dtype=np.float32)
                    last_text = ""


# ========================
# 7. main
# ========================
def main():
    root = tk.Tk()
    app = SubtitleApp(root)

    t1 = threading.Thread(target=audio_thread_func, daemon=True)
    t2 = threading.Thread(target=whisper_thread_func, daemon=True)
    t1.start()
    t2.start()

    root.mainloop()


if __name__ == "__main__":
    main()
