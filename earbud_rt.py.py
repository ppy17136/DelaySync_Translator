# earbud_rt.py  —— 实时“翻译耳机”模拟：Mic → ASR → MT → TTS，并向眼镜端发送 delay
import time, socket, json, argparse, queue, sys
from pathlib import Path

import numpy as np
import pyttsx3
import pyaudio
import webrtcvad

import whisper
from transformers import MarianMTModel, MarianTokenizer

from common import HOST, PORT  # 与 glasses_sim.py 保持一致

# =============== 音频采集（VAD 分段） ===============
class MicVAD:
    def __init__(self, rate=16000, frame_ms=30, vad_level=2, max_segment_s=8.0):
        self.rate = rate
        self.channels = 1
        self.width = 2  # 16-bit
        self.frame_bytes = int(rate * frame_ms / 1000) * self.width
        self.vad = webrtcvad.Vad(vad_level)  # 0~3 越大越敏感
        self.max_frames = int(max_segment_s * 1000 / frame_ms)

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                   channels=self.channels,
                                   rate=rate,
                                   input=True,
                                   frames_per_buffer=int(rate * frame_ms / 1000))

    def read_segment(self):
        """返回一段说话的 PCM bytes；静音返回 None"""
        frames = []
        voiced = False
        silence_count = 0
        # 先等待有声
        while True:
            buf = self.stream.read(int(self.frame_bytes / self.width), exception_on_overflow=False)
            if len(buf) == 0:
                return None
            is_speech = self.vad.is_speech(buf, self.rate)
            if is_speech:
                voiced = True
                frames.append(buf)
                break
        # 已进入有声段，持续读，直到静音 0.6s 或到达最大长度
        max_silence_frames = int(600 / 30)  # 600ms
        while len(frames) < self.max_frames:
            buf = self.stream.read(int(self.frame_bytes / self.width), exception_on_overflow=False)
            if len(buf) == 0:
                break
            is_speech = self.vad.is_speech(buf, self.rate)
            frames.append(buf)
            if is_speech:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >= max_silence_frames:
                    break
        pcm = b"".join(frames)
        return pcm if len(pcm) > 0 else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

# =============== ASR（Whisper 本地） ===============
class ASRWhisper:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)  # 可用 "small"/"medium" 提升准确率

    def transcribe_pcm16(self, pcm_bytes, sample_rate=16000, src_lang=None):
        # Whisper 接口需要 numpy float32
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        t0 = time.time()
        result = self.model.transcribe(audio, language=src_lang) if src_lang else self.model.transcribe(audio)
        t1 = time.time()
        text = (result.get("text") or "").strip()
        return text, (t1 - t0)

# =============== MT（MarianMT 本地） ===============
class MTLocal:
    def __init__(self, direction="zh-en"):
        if direction == "zh-en":
            model_name = "Helsinki-NLP/opus-mt-zh-en"
        elif direction == "en-zh":
            model_name = "Helsinki-NLP/opus-mt-en-zh"
        else:
            raise ValueError("Unsupported MT direction, use zh-en or en-zh")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, txt: str):
        if not txt:
            return ""
        t0 = time.time()
        batch = self.tokenizer([txt], return_tensors="pt", padding=True)
        gen = self.model.generate(**batch, max_new_tokens=128)
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        t1 = time.time()
        return out, (t1 - t0)

# =============== TTS（pyttsx3 本地） ===============
class TTSLocal:
    def __init__(self, voice_name=None, rate=180):
        self.engine = pyttsx3.init()
        if voice_name:
            # 可按需挑选系统里中文/英文音色
            for v in self.engine.getProperty('voices'):
                if voice_name.lower() in (v.name or "").lower():
                    self.engine.setProperty('voice', v.id)
                    break
        self.engine.setProperty('rate', rate)

    def speak_block(self, text: str):
        if not text:
            return 0.0
        t0 = time.time()
        self.engine.say(text)
        self.engine.runAndWait()
        t1 = time.time()
        return (t1 - t0)

# =============== 发送延迟参数给眼镜端 ===============
def send_delay(delay_ms: int, tts_start_at: float, host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {"type": "delay_update", "delay_ms": int(delay_ms), "tts_start_at": float(tts_start_at)}
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
    sock.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_lang", type=str, default=None, help="源语言提示，如 zh/en（可留空让 Whisper 自检）")
    ap.add_argument("--mt", type=str, default="zh-en", choices=["zh-en", "en-zh"], help="翻译方向")
    ap.add_argument("--whisper", type=str, default="base", help="Whisper 模型：tiny/base/small/medium/large")
    ap.add_argument("--host", type=str, default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--lead_ms", type=int, default=600, help="提早告知眼镜的提前量（毫秒）")
    args = ap.parse_args()

    mic = MicVAD()
    asr = ASRWhisper(args.whisper)
    mt  = MTLocal(args.mt)
    tts = TTSLocal()  # 如需中文女声可在 Windows 选择 'Huihui' 等

    print("[Earbud-RT] 就绪。开始说话（VAD 自动分段）。Ctrl+C 退出。")

    try:
        while True:
            pcm = mic.read_segment()
            if not pcm:
                continue

            # 1) ASR
            txt_asr, d_asr = asr.transcribe_pcm16(pcm, 16000, src_lang=args.src_lang)
            print(f"[ASR] {txt_asr}  ({d_asr:.2f}s)")

            # 2) MT
            txt_mt, d_mt = mt.translate(txt_asr)
            print(f"[MT ] {txt_mt}  ({d_mt:.2f}s)")

            # 3) 估计 TTS 时长（粗估：每字~80ms，可根据语速&标点修正）
            est_tts = max(0.6, 0.08 * max(1, len(txt_mt)))

            # 4) 告知眼镜端：我们打算在 tts_start_at 开始播报；眼镜需要准备 delay_ms 的视频时移
            delay_ms = int(1000 * (d_asr + d_mt + est_tts))
            tts_start_at = time.time() + args.lead_ms / 1000.0
            send_delay(delay_ms, tts_start_at, args.host, args.port)
            print(f"[SYNC] delay_ms={delay_ms}  tts_start_at={tts_start_at:.3f}  (lead={args.lead_ms}ms)")

            # 5) 等到 tts_start_at，进行 TTS 实播，并记录真实 TTS 用时，用于下次校正
            now = time.time()
            if tts_start_at > now:
                time.sleep(tts_start_at - now)

            d_tts = tts.speak_block(txt_mt)
            print(f"[TTS] done in {d_tts:.2f}s")

            # （可选）用真实 d_tts 校正下一次 est_tts 的经验参数
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        print("[Earbud-RT] 结束。")

if __name__ == "__main__":
    main()
