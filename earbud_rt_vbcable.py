# earbud_rt_vbcable.py
# Windows: 直连系统播放声音（通过 VB-Audio Virtual Cable）
# 流程：系统播放器输出 -> CABLE Input；本脚本从 CABLE Output 采集 -> ASR(Whisper) -> MT(Marian) -> TTS(pyttsx3)
# 并向眼镜端发送 delay_ms / tts_start_at 实现视频延时与翻译延时同步

import time, socket, json, argparse
from typing import Optional
import os, wave, tempfile

import numpy as np
import pyttsx3
import pyaudio
import webrtcvad
import whisper
from transformers import MarianMTModel, MarianTokenizer

from common import HOST, PORT  # 与 glasses_sim.py 保持一致


# ===================== 工具：列出/查找输入设备 =====================
def list_input_devices():
    pa = pyaudio.PyAudio()
    print("=== 可用输入设备（index, name） ===")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"[{i}] {info.get('name')}")
    pa.terminate()

def find_device_index_by_name_substring(name_substr: str) -> Optional[int]:
    """返回第一个名称包含子串的输入设备索引；找不到返回 None"""
    name_substr = (name_substr or "").lower()
    pa = pyaudio.PyAudio()
    idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            name = (info.get("name") or "").lower()
            if name_substr in name:
                idx = i
                break
    pa.terminate()
    return idx

#查找输出设备的工具
def find_output_device_index_by_name_substring(name_substr):
    import pyaudio
    name_substr = (name_substr or "").lower()
    pa = pyaudio.PyAudio()
    idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxOutputChannels", 0)) > 0:
            name = (info.get("name") or "").lower()
            if name_substr in name:
                idx = i
                break
    pa.terminate()
    return idx


# ===================== 音频采集（VAD 自动分段） =====================
class LineInVAD:
    """
    从指定输入设备采集（如 CABLE Output），用 WebRTC VAD 把语音自动切成“句段”。
    采样率 16k / 16-bit 单声道；frame_ms=30ms；静音 600ms 作为段尾。
    """
    def __init__(self, rate=16000, frame_ms=30, vad_level=2,
                 max_segment_s=12.0, device_index: Optional[int]=None):
        self.rate = rate
        self.channels = 1
        self.width = 2  # int16
        self.frame_samples = int(rate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(vad_level)  # 0~3，越大越“敏感”
        self.max_frames = int(max_segment_s * 1000 / frame_ms)

        self.pa = pyaudio.PyAudio()
        kwargs = dict(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=rate,
            input=True,
            frames_per_buffer=self.frame_samples,
        )
        if device_index is not None:
            kwargs["input_device_index"] = device_index

        self.stream = self.pa.open(**kwargs)

    def read_segment(self) -> Optional[bytes]:
        """阻塞式读取一段‘有声段’的 PCM16；若无数据返回 None"""
        frames = []

        # 等待进入有声
        while True:
            buf = self.stream.read(self.frame_samples, exception_on_overflow=False)
            if not buf:
                return None
            if self.vad.is_speech(buf, self.rate):
                frames.append(buf)
                break

        # 已进入有声段；直到静音 600ms 或达上限
        max_silence_frames = int(600 / 30)  # 600ms
        silence = 0
        while len(frames) < self.max_frames:
            buf = self.stream.read(self.frame_samples, exception_on_overflow=False)
            if not buf:
                break
            speech = self.vad.is_speech(buf, self.rate)
            frames.append(buf)
            if speech:
                silence = 0
            else:
                silence += 1
                if silence >= max_silence_frames:
                    break

        return b"".join(frames) if frames else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


# ===================== ASR（Whisper 本地） =====================
class ASRWhisper:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate=16000, src_lang=None):
        # Whisper 输入需要 float32 [-1,1]
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        t0 = time.time()
        if src_lang:
            result = self.model.transcribe(audio, language=src_lang)
        else:
            result = self.model.transcribe(audio)
        t1 = time.time()
        text = (result.get("text") or "").strip()
        return text, (t1 - t0)


# ===================== MT（MarianMT 本地） =====================
class MTLocal:
    def __init__(self, direction="en-zh"):
        if direction == "zh-en":
            model_name = "Helsinki-NLP/opus-mt-zh-en"
        elif direction == "en-zh":
            model_name = "Helsinki-NLP/opus-mt-en-zh"
        else:
            raise ValueError("Unsupported MT direction, choose zh-en or en-zh")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, txt: str):
        if not txt:
            return "", 0.0
        import torch
        t0 = time.time()
        batch = self.tokenizer([txt], return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = self.model.generate(**batch, max_new_tokens=128)
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        t1 = time.time()
        return out, (t1 - t0)


# ===================== TTS（pyttsx3 本地） =====================
#“可强制到指定输出设备”
class TTSLocal:
    def __init__(self, voice_name=None, rate=180,
                 out_device_name=None, out_device_index=None):
        self.voice_name = voice_name
        self.rate = rate
        self.force_device = (out_device_name is not None) or (out_device_index is not None)
        self.out_device_index = out_device_index
        if self.out_device_index is None and out_device_name:
            self.out_device_index = find_output_device_index_by_name_substring(out_device_name)

    def _new_engine(self):
        import pyttsx3
        eng = pyttsx3.init('sapi5')  # 明确使用 Windows SAPI5
        if self.voice_name:
            for v in eng.getProperty('voices'):
                n = (v.name or "") + " " + (v.id or "")
                if self.voice_name.lower() in n.lower():
                    eng.setProperty('voice', v.id)
                    break
        eng.setProperty('rate', self.rate)
        return eng

    def speak_block(self, text: str) -> float:
        if not text:
            return 0.0
        t0 = time.time()

        # 不强制设备：走系统默认（最省事）
        if not self.force_device or self.out_device_index is None:
            eng = self._new_engine()
            eng.say(text)
            eng.runAndWait()
            try: eng.stop()
            except: pass
            return time.time() - t0

        # 强制到指定输出设备：先生成 WAV，再用 PyAudio 播放到 out_device_index
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        eng = self._new_engine()
        eng.save_to_file(text, wav_path)
        eng.runAndWait()
        try: eng.stop()
        except: pass

        # 播放到指定输出设备（例如 CABLE Input）
        wf = wave.open(wav_path, 'rb')
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True,
                             output_device_index=self.out_device_index,
                             frames_per_buffer=1024)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()
            wf.close()
            try: os.remove(wav_path)
            except: pass

        return time.time() - t0




# ===================== 发送延迟参数给眼镜端 =====================
def send_delay(delay_ms: int, tts_start_at: float, host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {"type": "delay_update", "delay_ms": int(delay_ms), "tts_start_at": float(tts_start_at)}
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
    sock.close()


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser(description="Windows VB-CABLE 直连系统音频的实时翻译耳机端")
    ap.add_argument("--src_lang", type=str, default="en", help="源语言提示（如 en/zh；留空=自动检测）")
    ap.add_argument("--mt", type=str, default="en-zh", choices=["en-zh", "zh-en"], help="翻译方向")
    ap.add_argument("--whisper", type=str, default="base", help="Whisper 模型：tiny/base/small/medium/large")
    ap.add_argument("--lead_ms", type=int, default=800, help="提前告知眼镜的时间（毫秒）")
    ap.add_argument("--host", type=str, default=HOST, help="眼镜端 UDP 地址")
    ap.add_argument("--port", type=int, default=PORT, help="眼镜端 UDP 端口")
    ap.add_argument("--device-name", type=str, default="CABLE Output", help="输入设备名子串（默认匹配 VB-CABLE 的 CABLE Output）")
    ap.add_argument("--device-index", type=int, default=None, help="输入设备索引（优先于 device-name）")
    ap.add_argument("--vad-level", type=int, default=2, choices=[0,1,2,3], help="VAD 敏感度（高=更容易判定为有声）")
    ap.add_argument("--max-seg-s", type=float, default=12.0, help="单段最长秒数（上限保护）")
    ap.add_argument("--list-devices", action="store_true", help="仅列出可用输入设备并退出")
    ap.add_argument("--tts-to-cable", action="store_true",
                    help="将中文TTS强制播放到指定输出设备（默认 CABLE Input）")
    ap.add_argument("--tts-device-name", type=str, default="CABLE Input",
                    help="TTS 指定输出设备名称子串（默认 CABLE Input）")
    ap.add_argument("--tts-device-index", type=int, default=None,
                    help="TTS 指定输出设备索引（优先级高于名称）")

    args = ap.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    # 解析输入设备
    device_index = args.device_index
    if device_index is None and args.device_name:
        device_index = find_device_index_by_name_substring(args.device_name)
        if device_index is None:
            print(f"[Warn] 未找到包含 '{args.device_name}' 的输入设备。可用 --list-devices 查看并用 --device-index 指定。")
            list_input_devices()
            return
        else:
            print(f"[Info] 使用输入设备 index={device_index}（匹配到名称子串：{args.device_name}）")

    # 初始化各模块
    mic = LineInVAD(device_index=device_index, vad_level=args.vad_level, max_segment_s=args.max_seg_s)
    asr = ASRWhisper(args.whisper)
    mt  = MTLocal(args.mt)
    #tts = TTSLocal()  # 如需指定中文女声（Win），可安装中文语音包后传 voice_name 参数
    tts = TTSLocal(
        voice_name=None, rate=180,
        out_device_name=(args.tts_device_name if args.tts_to_cable else None),
        out_device_index=(args.tts_device_index if args.tts_to_cable else None)
    )


    print("[Earbud-RT-VBCABLE] 就绪。请将系统/播放器输出设为『CABLE Input』。Ctrl+C 结束。")

    try:
        while True:
            # 读一段有声
            pcm = mic.read_segment()
            if not pcm:
                continue

            # 1) ASR（英文）
            txt_src, d_asr = asr.transcribe_pcm16(pcm, 16000, src_lang=args.src_lang)
            print(f"[ASR] {txt_src}  ({d_asr:.2f}s)")
            if not txt_src:
                continue

            # 2) MT（英->中 或 中->英）
            txt_tgt, d_mt = mt.translate(txt_src)
            print(f"[MT ] {txt_tgt}  ({d_mt:.2f}s)")

            # 3) 估计 TTS 用时（经验：每个汉字/英文 token ~0.08s；至少 0.6s）
            est_tts = max(0.6, 0.08 * max(1, len(txt_tgt)))

            # 4) 告知眼镜端：delay_ms & tts_start_at
            delay_ms = int(1000 * (d_asr + d_mt + est_tts))
            tts_start_at = time.time() + args.lead_ms / 1000.0
            send_delay(delay_ms, tts_start_at, args.host, args.port)
            print(f"[SYNC] delay_ms={delay_ms}  tts_start_at={tts_start_at:.3f}  (lead={args.lead_ms}ms)")

            # 5) 到点播报中文 TTS（阻塞播放）
            now = time.time()
            if tts_start_at > now:
                time.sleep(tts_start_at - now)
            d_tts = tts.speak_block(txt_tgt)
            print(f"[TTS] done in {d_tts:.2f}s")

            # （可选）可用真实 d_tts 做自校正：
            # est_tts = 0.2 * d_tts + 0.8 * est_tts

    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        print("[Earbud-RT-VBCABLE] 结束。")


if __name__ == "__main__":
    main()

