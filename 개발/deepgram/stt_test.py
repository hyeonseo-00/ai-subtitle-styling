import os
import asyncio
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent

load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")


async def main():

    if not API_KEY:
        print("❌ DEEPGRAM_API_KEY 없음")
        return

    dg = AsyncDeepgramClient(api_key=API_KEY)

    # 이벤트 핸들러
    def on_message(message):
        if isinstance(message, ListenV1ResultsEvent):
            if message.channel and message.channel.alternatives:
                transcript = message.channel.alternatives[0].transcript
                if transcript:
                    print("🗣️", transcript)

    def on_error(error):
        print("❌ 오류:", error)

    def on_open(_):
        print("✅ 연결 성공! 마이크 입력 시작")

    def on_close(_):
        print("👋 연결 종료")

    print("🔌 Deepgram 연결 중...")

    # v1 API 사용
    async with dg.listen.v1.connect(
        model="nova-2",
        language="ko-KR",
        encoding="linear16",
        sample_rate="16000",
        smart_format="true"
    ) as connection:
        # 이벤트 등록
        connection.on(EventType.OPEN, on_open)
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)
        connection.on(EventType.CLOSE, on_close)

        # 비동기로 메시지 수신 시작
        listen_task = asyncio.create_task(connection.start_listening())

        # PyAudio 사용
        import pyaudio

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )

        try:
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                await connection.send_media(data)
                await asyncio.sleep(0.01)  # 작은 딜레이 추가
        except KeyboardInterrupt:
            print("\n👋 종료 중...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            listen_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 종료")
