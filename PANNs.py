import os
import sys
import urllib.request
import ssl
import cv2
import numpy as np
import sounddevice as sd
import time
import imageio_ffmpeg as ffmpeg_bin
import subprocess
from PIL import ImageFont, ImageDraw, Image
import torch
import librosa
from contextlib import contextmanager

# ==========================================
# [0] 로그 차단기
# ==========================================
@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            try:
                fd_stderr = 2
                fd_dup = os.dup(fd_stderr)
                os.dup2(devnull.fileno(), fd_stderr)
                yield
            except Exception:
                yield
            finally:
                try:
                    os.dup2(fd_dup, fd_stderr)
                    os.close(fd_dup)
                except Exception:
                    pass
        finally:
            sys.stderr = old_stderr

# ==========================================
# [1] 시스템 설정
# ==========================================
def check_panns_setup():
    ssl._create_default_https_context = ssl._create_unverified_context
    home_dir = os.path.expanduser("~")
    panns_dir = os.path.join(home_dir, "panns_data")
    if not os.path.exists(panns_dir): os.makedirs(panns_dir)
    
    csv_path = os.path.join(panns_dir, "class_labels_indices.csv")
    if not os.path.exists(csv_path): pass 

    model_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    model_path = os.path.join(panns_dir, "Cnn14_mAP=0.431.pth")
    if not os.path.exists(model_path):
        try: urllib.request.urlretrieve(model_url, model_path)
        except Exception: sys.exit(1)

check_panns_setup()
from panns_inference import AudioTagging

# ==========================================
# [2] 설정
# ==========================================
VIDEO_PATH = './outputs/펜트_video.mp4'
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
SAMPLE_RATE = 32000
VOLUME_BOOST = 6.0 
ANALYSIS_INTERVAL = 0.1
BGM_HOLD_TIME = 2.0 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> 사용 장치: {device}")

with suppress_stderr():
    model = AudioTagging(checkpoint_path=None, device=device)
labels = model.labels

# ==========================================
# [3] 번역 사전
# ==========================================
translation_dict = {
    # --- [A] 배경음악 ---
    'Dramatic music': '웅장한 음악이 흐른다', 'Film score': '영화 같은 웅장한 선율이 깔린다',
    'Orchestra': '오케스트라 연주가 시작된다', 'Choir': '웅장한 합창 소리가 울린다',
    'Soundtrack music': '비장한 음악이 흐른다', 'Theme music': '테마곡이 흐른다',
    'Symphony': '웅장한 교향곡이 흐른다',

    'Sad music': '슬픈 선율이 흐른다', 
    'Lullaby': '잔잔한 자장가가 들린다', 'Music': '잔잔한 배경음악이 흐른다', 
    'Background music': '배경음악이 깔린다',

    'Happy music': '경쾌한 음악이 흐른다', 'Exciting music': '박진감 넘치는 음악이 흐른다',
    'Pop music': '신나는 팝송이 나온다', 'Rock music': '강렬한 락 음악이 터져 나온다',
    'Electronic music': '신나는 전자음이 들린다', 'Hip hop music': '힙합 비트가 흐른다',
    'Disco': '신나는 디스코 음악이 나온다',

    'Scary music': '으스스한 음악이 흐른다', 'Suspense': '긴장감 넘치는 음악이 흐른다',

    # --- [B] 사람 소리 ---
    'Breathing': '거친 숨소리가 들린다', 'Pant': '숨을 헐떡인다', 'Gasp': '숨을 들이킨다',
    'Sigh': '깊은 한숨을 내쉰다', 'Throat clearing': '목을 가다듬는다',
    'Cough': '콜록거리며 기침을 한다', 'Sneeze': '재채기를 한다', 
    'Screaming': '비명 소리가 울려 퍼진다', 'Crying, sobbing': '누군가 흐느껴 운다', 
    'Laughter': '웃음 소리가 들린다', 'Footsteps': '발자국 소리가 들린다', 
    'Crowd': '사람들이 웅성거린다',

    # --- [C] 전투/액션 ---
    'Punch': '둔탁한 주먹 소리가 난다',
    'Slap, smack': '짝! (때리는 소리)',
    'Clapping': '짝! (박수 소리)',     
    'Applause': '박수 갈채가 쏟아진다', 
    
    'Thump, thud': '쿵! 하고 부딪힌다', 'Fighting': '격한 몸싸움 소리가 들린다', 
    'Wrestling': '옷깃이 스치며 뒤엉킨다', 'Whoosh, swoosh, swish': '무언가 휙 하고 지나간다',
    'Clang': '날카로운 칼 부딪히는 소리가 난다',
    'Gunshot, gunfire': '총성이 울린다', 
    'Explosion': '거대한 폭발음이 들린다',
    'Tools': '철그럭거리는 소리가 난다',

    # --- [D] 유리 소리 (오인식 주의) ---
    'Shatter': '유리가 와장창 깨진다', 
    'Glass': '유리가 깨지는 소리가 난다',

    # --- [E] 자연/사물 ---
    'Rain': '빗소리가 들린다', 'Thunder': '천둥이 친다', 'Wind': '바람이 세차게 분다', 
    'Water': '물 흐르는 소리가 들린다', 'Fire': '불이 타오르는 소리가 난다',
    'Door': '문이 열리는 소리가 난다', 'Knock': '누군가 문을 두드린다'
}

def get_korean_label(english_label):
    return translation_dict.get(english_label, None)

# ==========================================
# [4] 키워드 분류 리스트
# ==========================================
ALL_MUSIC_KEYS = [
    'Dramatic music', 'Film score', 'Orchestra', 'Choir', 'Soundtrack music', 'Theme music',
    'Sad music', 'Tender music', 'Lullaby', 'Happy music', 'Exciting music', 'Pop music', 
    'Rock music', 'Electronic music', 'Disco', 'Hip hop music', 'Scary music', 'Suspense',
    'Music', 'Background music', 'Musical instrument', 'Plucked string instrument',
    'Piano', 'Guitar', 'Electric guitar', 'Bass guitar', 'Acoustic guitar', 
    'Violin, fiddle', 'Cello', 'Harp', 'Synthesizer', 'Drum kit', 'Drum',
    'Brass instrument', 'Woodwind instrument', 'Percussion', 'Keyboard (musical)'
]

PRIORITY_GENRE_KEYS = [
    'Rock music', 'Pop music', 'Hip hop music', 'Electronic music', 'Disco',
    'Dramatic music', 'Film score', 'Orchestra', 'Soundtrack music',
    'Sad music', 'Tender music', 'Scary music', 'Suspense', 'Happy music', 'Exciting music'
]

# ★ 뺨 때리는 소리 (민감도 최상)
SLAP_KEYS = ['Slap, smack', 'Clapping', 'Hands'] 

# ★ 유리 소리 (민감도 최하 - 뺨소리 오인식 방지용)
GLASS_KEYS = ['Glass', 'Shatter', 'Breaking']

FIGHT_KEYS = [
    'Thump, thud', 'Punch', 'Wrestling', 'Fighting', 'Grunt', 'Groan',
    'Smash, crash', 'Whack, thwack', 'Whoosh, swoosh, swish', 'Clang', 'Ding', 
    'Metal', 'Explosion', 'Gunshot, gunfire', 
    'Cutlery, silverware', 'Dishes, pots, and pans', 'Tools', 'Hammer', 'Mechanisms'
]

GENERAL_SFX_KEYS = [
    'Breathing', 'Pant', 'Gasp', 'Sigh', 'Throat clearing', 'Cough', 'Sneeze',
    'Screaming', 'Crying, sobbing', 'Laughter', 'Footsteps', 'Applause', 'Cheering', 'Crowd',
    'Rain', 'Thunder', 'Wind', 'Water', 'Fire', 'Door', 'Knock'
]

SFX_KEYS = FIGHT_KEYS + GENERAL_SFX_KEYS + SLAP_KEYS + GLASS_KEYS

IGNORE_LIST = [
    'Metal','Cutlery, silverware','Tender music', 'Cheering',
    'Silence', 'Speech', 'Male speech, man speaking', 'Female speech, woman speaking', 
    'Child speech, kid speaking', 'Conversation', 'Narration, monologue', 'Babbling', 
    'Inside, small room', 'Inside, large room', 'Outside, urban, or man-made', 
    'Static', 'Noise', 'White noise', 'Pink noise', 'Ambience'
]

AMBIGUOUS_SFX = ['Thump, thud', 'Smash, crash', 'Whack, thwack', 'Clang', 'Metal', 'Ding', 'Tools', 'Hammer', 'Mechanisms', 'Scrape', 'Rub', 'Noise']

# ==========================================
# [5] 유틸리티
# ==========================================
def extract_audio_ffmpeg(v_path, a_path):
    if os.path.exists(a_path): os.remove(a_path)
    cmd = [ffmpeg_bin.get_ffmpeg_exe(), "-i", v_path, "-vn", "-acodec", "pcm_s16le", 
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", a_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def put_dual_text(image, bgm_text, sfx_text, frame_width, frame_height):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype(FONT_PATH, max(20, int(frame_width / 35)))
    except: font = ImageFont.load_default()
    
    center_x = frame_width // 2
    bottom_margin = int(frame_height * 0.9) 
    
    if bgm_text:
        text = f"♪ {bgm_text}"
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = center_x - w // 2, bottom_margin - h - 10
        draw.rectangle([(x-10, y-5), (x+w+10, y+h+5)], fill=(0,0,0,160))
        draw.text((x, y), text, font=font, fill=(100, 255, 255)) 
        bottom_margin = y - 15 

    if sfx_text:
        fight_flags = ['퍽', '쿵', '쾅', '탕', '총', '폭발', '짝', '때리는']
        if any(k in sfx_text for k in fight_flags): color = (255, 80, 80)
        else: color = (255, 255, 100)
        
        text = f"🔊 {sfx_text}"
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = center_x - w // 2, bottom_margin - h - 10
        draw.rectangle([(x-10, y-5), (x+w+10, y+h+5)], fill=(0,0,0,160))
        draw.text((x, y), text, font=font, fill=color)

    return np.array(img_pil)

# ==========================================
# [6] 메인 실행
# ==========================================
def main():
    if not os.path.exists(VIDEO_PATH): 
        print(f"파일을 찾을 수 없습니다: {VIDEO_PATH}")
        return

    temp_audio = 'temp_audio_final.wav'
    print("[진행] 오디오 추출 중...")
    extract_audio_ffmpeg(VIDEO_PATH, temp_audio)
    
    audio_origin, _ = librosa.load(temp_audio, sr=SAMPLE_RATE)
    audio_for_ai = audio_origin * VOLUME_BOOST 
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("[진행] 재생 시작 (종료: q)")
    sd.play(audio_origin, SAMPLE_RATE)
    start_time = time.time()

    # 상태 관리 변수
    current_bgm_display = ""
    current_sfx_display = ""
    last_printed_bgm = "" 
    bgm_last_detected_time = 0 
    last_detected_bgm_text = "" 
    last_pred_time = 0
    prev_rms = 0.0

    with suppress_stderr():
        while cap.isOpened():
            elapsed = time.time() - start_time
            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000)
            ret, frame = cap.read()
            if not ret: break

            if elapsed - last_pred_time > ANALYSIS_INTERVAL:
                idx = int(elapsed * SAMPLE_RATE)
                short_window = int(SAMPLE_RATE * 0.3) 
                start_idx = max(0, idx - short_window)
                
                waveform_seg = audio_for_ai[start_idx:idx]
                
                if len(waveform_seg) > 500:
                    rms = np.sqrt(np.mean(waveform_seg**2))
                    is_impact = (rms > prev_rms * 1.5) or (rms > 0.1)
                    prev_rms = rms

                    target_len = SAMPLE_RATE
                    repeats = (target_len // len(waveform_seg)) + 1
                    tiled_seg = np.tile(waveform_seg, repeats)[:target_len]
                    
                    with torch.no_grad():
                        output, _ = model.inference(tiled_seg[None, :])
                    
                    scores = output[0]
                    top_idx = np.argsort(scores)[::-1] 
                    
                    bgm_candidates = []
                    sfx_candidates_raw = []

                    # 상위 5개 탐색
                    for i in top_idx[:5]:
                        label = labels[i]
                        score = scores[i]
                        
                        if label in IGNORE_LIST: continue
                        korean = get_korean_label(label)
                        if not korean: continue

                        # BGM
                        if label in ALL_MUSIC_KEYS:
                            min_score = 0.6 if label not in PRIORITY_GENRE_KEYS and label != 'Music' else 0.05
                            if score > min_score:
                                bgm_candidates.append((label, score, korean))
                        
                        # SFX
                        elif label in SFX_KEYS:
                            # ★ [핵심] 임계값 차별화 ★
                            if label in GLASS_KEYS:
                                if score > 0.7:
                                    # 진짜 유리 소리
                                    thr = 0.7
                                elif is_impact and score > 0.15:
                                    # 유리 점수는 낮지만 충격이 있다 -> 뺨 소리로 강제 변환
                                    korean = '짝! (때리는 소리)' # 라벨 바꿔치기
                                    thr = 0.15
                                else:
                                    thr = 0.7 # 무시
                            
                            elif label in SLAP_KEYS:
                                thr = 0.05 if is_impact else 0.25
                            elif label == 'Sneeze':
                                thr = 0.9
                            elif label == 'Gunshot, gunfire':
                                thr = 0.7 
                            elif label in FIGHT_KEYS and is_impact:
                                thr = 0.05
                            else:
                                thr = 0.3

                            if score > thr:
                                sfx_candidates_raw.append((label, korean))

                    # BGM 선정
                    temp_bgm = ""
                    if bgm_candidates:
                        genre_matches = [x for x in bgm_candidates if x[0] in PRIORITY_GENRE_KEYS]
                        if genre_matches:
                            genre_matches.sort(key=lambda x: x[1], reverse=True)
                            temp_bgm = genre_matches[0][2]
                        else:
                            bgm_candidates.sort(key=lambda x: x[1], reverse=True)
                            temp_bgm = bgm_candidates[0][2]
                        
                        bgm_last_detected_time = elapsed
                        last_detected_bgm_text = temp_bgm
                    else:
                        if elapsed - bgm_last_detected_time < BGM_HOLD_TIME:
                            temp_bgm = last_detected_bgm_text
                        else:
                            temp_bgm = ""

                    # SFX 선정
                    is_music_playing = (temp_bgm != "")
                    valid_sfx_list = []
                    for label_eng, label_ko in sfx_candidates_raw:
                        if is_music_playing and label_eng in AMBIGUOUS_SFX: continue
                        valid_sfx_list.append(label_ko)
                    
                    final_sfx = valid_sfx_list[0] if valid_sfx_list else ""

                    current_bgm_display = temp_bgm
                    current_sfx_display = final_sfx
                    
                    # 콘솔 출력 (중복 제거)
                    should_print = False
                    if temp_bgm != last_printed_bgm:
                        should_print = True
                        last_printed_bgm = temp_bgm
                    if final_sfx:
                        should_print = True

                    if should_print:
                        log_msg = f"[{elapsed:.1f}s] "
                        if temp_bgm: log_msg += f"BGM: {temp_bgm} "
                        if final_sfx: log_msg += f"| SFX: {final_sfx}"
                        if log_msg.strip() != f"[{elapsed:.1f}s]":
                            sys.stdout.write(log_msg + "\n")

                    last_pred_time = elapsed

            frame = put_dual_text(frame, current_bgm_display, current_sfx_display, width, height)
            cv2.imshow('Final Corrected Slap Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    sd.stop()
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_audio): os.remove(temp_audio)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stdout.write(f"Error: {e}\n")
        sd.stop()
        cv2.destroyAllWindows()