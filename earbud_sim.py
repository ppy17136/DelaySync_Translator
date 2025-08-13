# earbud_sim.py
import socket, json, time, argparse
import pyttsx3
from common import HOST, PORT, DEMO_TTS_TEXT

def send_delay(delay_ms: int, host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {
        "type": "delay_update",
        "delay_ms": int(delay_ms),
        # 发送一个“将要开始播放TTS的时间点”，给眼镜预缓冲参考
        "tts_start_at": time.time() + 0.6  # 留 0.6s 传输/处理余量
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
    sock.close()

def speak_tts(text: str):
    engine = pyttsx3.init()
    # 可按需调语速（默认大概 200 wpm）
    # engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay_ms", type=int, default=1200, help="翻译总延迟（毫秒），将用于视频端的延时显示")
    ap.add_argument("--host", type=str, default=HOST, help="眼镜模拟端的地址")
    ap.add_argument("--port", type=int, default=PORT, help="眼镜模拟端的UDP端口")
    ap.add_argument("--text", type=str, default=DEMO_TTS_TEXT, help="要播放的翻译语音文本")
    args = ap.parse_args()

    # 先把延迟通知给“眼镜”，让对方开始缓冲
    print(f"[Earbud] 即将发送延迟值 {args.delay_ms} ms 给眼镜，并准备播放TTS...")
    send_delay(args.delay_ms, args.host, args.port)

    # 等一小会儿，让眼镜端收到并进入缓冲
    time.sleep(0.6)

    t0 = time.time()
    speak_tts(args.text)
    t1 = time.time()
    print(f"[Earbud] TTS播放完毕，用时 {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
