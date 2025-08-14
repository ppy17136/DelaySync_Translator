# prosody_tts_vm.py
# 方案B：Voicemeeter 混音（播放器→CABLE-A；ASR←CABLE-A Output；TTS→Voicemeeter Input）
# 重点修复与优化：
# - Whisper CUDA 自适应（有就用 cuda+fp16，无则 cpu+fp32）
# - VAD 参数更稳（少切碎），并支持缓冲拼句 + 超时冲洗，显著减少“丢句”
# - edge-tts 使用合法的 rate/pitch 形式（±N% / ±NHz），默认走纯文本参数，不再读出 SSML
# - WAV 合成后强校验（RIFF/PCM、时长、能量），失败自动重试一次
# - 播放到指定输出设备前打印设备与音频格式；支持 --keep_wav / --wav_dir
# - make_out_path 修正（不再把临时文件名当目录）
# - 离线分支变量修正；重复过滤 & 简单幻听过滤
#
# 运行示例（直播）：
#   python prosody_tts_vm.py --mode live \
#     --device-name "CABLE-A Output" \
#     --tts_device_name "Voicemeeter Input (VB-Audio Voicemeeter VAIO)" \
#     --whisper small.en --lead_ms 1200 --voice zh-CN-YunxiNeural \
#     --keep_wav --wav_dir D:\vm_out
#
# 语速偏慢可以加： --tts_rate_pct -15   或   --rate_scale 0.7
# 若想更准可换模型： --whisper medium.en  （代价是更慢）

import argparse, asyncio, json, socket, time, os, wave, tempfile, re, math, struct, datetime
from typing import Optional, List, Tuple
import sounddevice as sd
import numpy as np
import pyaudio
import webrtcvad
import librosa
import soundfile as sf

import whisper
from transformers import MarianMTModel, MarianTokenizer
import edge_tts
from xml.sax.saxutils import escape as xml_escape

# 你的项目里有 common.py，保持一致即可
try:
    from common import HOST, PORT
except Exception:
    HOST, PORT = "127.0.0.1", 17800


# ================= UDP 同步（眼镜端） =================
def send_delay(delay_ms: int, tts_start_at: float, host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {"type": "delay_update", "delay_ms": int(delay_ms), "tts_start_at": float(tts_start_at)}
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
    finally:
        sock.close()


# ================= 设备工具 =================
def list_input_devices():
    pa = pyaudio.PyAudio()
    print("=== 可用输入设备（index, name）===")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"[{i}] {info.get('name')}")
    pa.terminate()

def find_device_index_by_name(name_substr: str, want_output=False) -> Optional[int]:
    name_substr = (name_substr or "").lower()
    pa = pyaudio.PyAudio()
    target = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        ch = int(info.get("maxOutputChannels" if want_output else "maxInputChannels", 0))
        if ch > 0:
            name = (info.get("name") or "").lower()
            if name_substr in name:
                target = i
                break
    pa.terminate()
    return target


# ================= 重采样（无 resampy 依赖） =================
def resample_to_48k(y: np.ndarray, sr: int):
    target_sr = 48000
    if sr == target_sr:
        return y.astype(np.float32), sr
    try:
        from scipy.signal import resample_poly
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        y2 = resample_poly(y, up, down).astype(np.float32)
        return y2, target_sr
    except Exception:
        n_out = int(round(len(y) * target_sr / sr))
        if n_out <= 1 or len(y) <= 1:
            return y.astype(np.float32), target_sr
        x = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        xi = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y2 = np.interp(xi, x, y).astype(np.float32)
        return y2, target_sr


# ================= 实时采集（VAD 分段）/ 读取 wav =================
class LineInVAD:
    def __init__(self, rate=16000, frame_ms=30, vad_level=2,
                 max_segment_s=8.0, silence_ms=600, device_index: Optional[int]=None):
        self.rate = rate
        self.frame_ms = frame_ms
        self.frame_samples = int(rate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(vad_level)
        self.max_frames = int(max_segment_s * 1000 / frame_ms)
        self.silence_frames = max(1, int(silence_ms / frame_ms))

        self.pa = pyaudio.PyAudio()
        self._kwargs = dict(format=pyaudio.paInt16, channels=1, rate=rate,
                            input=True, frames_per_buffer=self.frame_samples)
        if device_index is not None:
            self._kwargs["input_device_index"] = device_index
        self.stream = None
        self._open_stream()

    def _open_stream(self):
        try:
            if self.stream:
                self.stream.stop_stream(); self.stream.close()
        except Exception:
            pass
        self.stream = self.pa.open(**self._kwargs)

    def _safe_read(self) -> Optional[bytes]:
        for _ in range(2):
            try:
                return self.stream.read(self.frame_samples, exception_on_overflow=False)
            except Exception:
                try: self._open_stream()
                except Exception: time.sleep(self.frame_ms/1000.0)
        return None

    def read_segment(self) -> Optional[bytes]:
        frames = []
        # 等待进入有声
        while True:
            buf = self._safe_read()
            if not buf: return None
            if self.vad.is_speech(buf, self.rate):
                frames.append(buf); break
        # 连续收，直到静音 / 上限
        silence = 0
        while len(frames) < self.max_frames:
            buf = self._safe_read()
            if not buf: break
            speech = self.vad.is_speech(buf, self.rate)
            frames.append(buf)
            if speech: silence = 0
            else:
                silence += 1
                if silence >= self.silence_frames: break
        return b"".join(frames)

    def drain(self, seconds: float = 0.5):
        if not self.stream: return
        n = int((seconds * self.rate) / self.frame_samples)
        for _ in range(max(1, n)):
            try: self.stream.read(self.frame_samples, exception_on_overflow=False)
            except Exception:
                try: self._open_stream()
                except Exception: break

    def close(self):
        try:
            if self.stream:
                self.stream.stop_stream(); self.stream.close()
        finally:
            self.pa.terminate()


def wav_segments_from_file(wav_path: str, rate=16000, seg_len=4.0) -> List[np.ndarray]:
    y, sr = librosa.load(wav_path, sr=rate, mono=True)
    hop = int(seg_len * sr)
    segs = []
    for i in range(0, len(y), hop):
        chunk = y[i:i+hop]
        if len(chunk) < int(0.8*hop): break
        segs.append(chunk)
    return segs


# ================= ASR（Whisper 英语） =================
class ASRWhisper:
    def __init__(
        self,
        model_name="small.en",
        no_speech_threshold=0.60,
        logprob_threshold=-1.2,
        compression_ratio_threshold=2.6,
        temperature=(0.0, 0.2),
    ):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = (device == "cuda")
        self.model = whisper.load_model(model_name, device=device)
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.temperature = temperature

    def _transcribe_np(self, audio: np.ndarray, lang="en") -> Tuple[str, float, dict]:
        t0 = time.time()
        result = self.model.transcribe(
            audio.astype(np.float32),
            language=lang,
            temperature=self.temperature,
            no_speech_threshold=self.no_speech_threshold,
            logprob_threshold=self.logprob_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            condition_on_previous_text=True,  # 连贯上下文
            fp16=self.fp16,
            verbose=False,
        )
        t1 = time.time()
        text = (result.get("text") or "").strip()
        return text, (t1-t0), result

    def transcribe_pcm16(self, pcm_bytes: bytes, sr=16000, lang="en") -> Tuple[str, float]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        txt, dur, _ = self._transcribe_np(audio, lang=lang)
        return txt, dur

    def transcribe_float(self, y: np.ndarray, lang="en") -> Tuple[str, float]:
        txt, dur, _ = self._transcribe_np(y.astype(np.float32), lang=lang)
        return txt, dur


# ================= MT（Marian 英->中） =================
class MTMarian:
    def __init__(self, direction="en-zh"):
        name = "Helsinki-NLP/opus-mt-en-zh" if direction=="en-zh" else "Helsinki-NLP/opus-mt-zh-en"
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name)

    def translate(self, txt: str) -> Tuple[str,float]:
        if not txt: return "", 0.0
        import torch
        t0 = time.time()
        batch = self.tokenizer([txt], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=256)
        zh = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        t1 = time.time()
        return zh, (t1-t0)


# ================= 韵律分析 & 参数映射 =================
def prosody_from_audio(y: np.ndarray, sr: int) -> dict:
    rmse = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    peaks = (rmse > (rmse.mean() + 0.5*rmse.std())).astype(np.int32)
    syll_rate = peaks.sum() / (len(y)/sr + 1e-6)
    try:
        f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, frame_length=2048, hop_length=256)
        f0 = f0[np.isfinite(f0)]
        mean_f0 = float(np.median(f0)) if f0.size else 130.0
    except Exception:
        mean_f0 = 130.0
    tail = y[-sr//2:] if len(y)>sr//2 else y
    tail_rms = float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.0
    pause = 0.2 if tail_rms < 0.02 else 0.0
    return {"syll_rate": float(syll_rate), "mean_f0": mean_f0, "pause_tail": pause}

def map_prosody_to_params(rate_syll: float, mean_f0: float,
                          base_rate_pct: int = -5, rate_scale: float = 1.0,
                          pitch_scale: float = 1.0) -> Tuple[str,str]:
    """
    返回适配 edge-tts 的 rate/pitch 字符串： "+10%" / "-3Hz"
    """
    # syll_rate 约 2~6：线性映射到 -10% ~ +30% 再乘以 rate_scale
    # 再叠加用户的基准覆盖 base_rate_pct
    norm = np.clip((rate_syll - 3.0) / 2.0, -1.0, 1.5)  # -1..1.5
    dyn_pct = int(np.clip( (norm * 20.0) * rate_scale, -30, +40 ))
    rate_pct = int(np.clip(base_rate_pct + dyn_pct, -50, +50))
    rate = f"{'+' if rate_pct>=0 else ''}{rate_pct}%"

    # mean_f0 相对 130Hz 的偏移，映射到 ±8Hz，乘以 pitch_scale
    dev = np.clip((mean_f0 - 130.0) / 130.0, -0.3, 0.3)
    hz = int(np.clip(dev * 25.0 * pitch_scale, -12, +12))
    pitch = f"{'+' if hz>=0 else ''}{hz}Hz"
    return rate, pitch


# ================= 文本清洗/重复过滤 =================
_PUNCT_RE = re.compile(r"^[\s\.,!?;:'\"“”‘’\-\(\)\[\]…·、—0-9]+$")

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'[\[\(（].*?[\]\)）]', ' ', text)
    text = re.sub(r'[<>]{1,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_trivial(txt: str) -> bool:
    if not txt: return True
    if _PUNCT_RE.match(txt): return True
    letters = sum(ch.isalpha() for ch in txt)
    return letters < 3


# ================= TTS（文本 → MP3 → WAV@48kHz） =================
async def edge_tts_to_wav(text: str, out_wav: str, voice: str,
                          rate: Optional[str]=None, pitch: Optional[str]=None) -> float:
    """
    用 edge-tts 合成，text 走纯文本（不使用 SSML），并给定合法的 rate/pitch（±N% / ±NHz）
    写 WAV 并返回时长（秒）
    """
    kwargs = {"voice": voice}
    if rate:  kwargs["rate"]  = rate   # 例如 "+10%"
    if pitch: kwargs["pitch"] = pitch  # 例如 "-3Hz"
    communicate = edge_tts.Communicate(text, **kwargs)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        mp3_path = tmp.name
    with open(mp3_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

    try:
        y, sr = librosa.load(mp3_path, sr=None, mono=True)
    except Exception:
        y, sr = librosa.load(mp3_path, sr=24000, mono=True)
    y, sr = resample_to_48k(y, sr)
    sf.write(out_wav, y, sr, subtype="PCM_16")

    # 强校验 + 一次重试
    try:
        ch, sr2, nframes = assert_valid_wav(out_wav)
    except Exception:
        sf.write(out_wav, y, sr, subtype="PCM_16")
        ch, sr2, nframes = assert_valid_wav(out_wav)

    try: os.remove(mp3_path)
    except Exception: pass
    return float(nframes / sr2)


# —— 工具：WAV 有效性校验 + 轻量能量检查
def assert_valid_wav(path: str) -> Tuple[int, int, int]:
    """返回 (channels, samplerate, nframes)。若无效抛异常。"""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise RuntimeError("not a RIFF/WAVE file")
        audio_fmt = None
        ch = sr = nframes = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                break
            cid, csz = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            data = f.read(csz)
            if cid == b"fmt ":
                if csz >= 16:
                    audio_fmt, ch, sr = struct.unpack("<HHI", data[:8])
            elif cid == b"data":
                if audio_fmt is None or ch is None:
                    raise RuntimeError("fmt chunk before data not parsed")
                bytes_per_samp = 2 if audio_fmt == 1 else 4
                nframes = len(data) // (bytes_per_samp * ch)
                break
        if nframes is None or sr is None or ch is None:
            raise RuntimeError("wav header incomplete")
        if os.path.getsize(path) <= 44:
            raise RuntimeError("wav file too small")
        return ch, sr, nframes

def verify_wav_ok(wav_path: str, min_dur: float = 0.05, min_rms: float = 1e-4) -> Tuple[bool, float]:
    if not os.path.exists(wav_path):
        return False, 0.0
    try:
        with open(wav_path, "rb") as f:
            if f.read(4) != b"RIFF":
                return False, 0.0
        with wave.open(wav_path, "rb") as wf:
            n = wf.getnframes()
            sr = wf.getframerate()
            ch = wf.getnchannels()
            if n <= 0 or sr <= 0:
                return False, 0.0
            dur = n / float(sr)
            if dur < min_dur:
                return False, dur
        frames_to_read = int(min(0.2, dur) * sr)
        with wave.open(wav_path, "rb") as wf:
            raw = wf.readframes(frames_to_read)
        if raw:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
            if rms < min_rms:
                return False, dur
        return True, dur
    except Exception:
        return False, 0.0


# ================= 播放到指定输出设备（Voicemeeter Input） =================
def play_wav_to_device(wav_path: str, out_device_index: Optional[int] = None):
    """
    用 sounddevice 直接把 WAV 播到指定输出设备。
    - 强制 48kHz（你的 TTS 那边已经是 48k）
    - 1 声道会自动“复制成 2 声道”，让 Voicemeeter 虚拟输入的电平条更明显
    - 阻塞播放，直到播完（和你之前的行为一致）
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    # 读 WAV（float32, 二维）
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (N, C)
    ch = data.shape[1]
    print(f"[PLAY] (sd) 格式检查: {ch}ch @ {sr}Hz, path={wav_path}")

    # 若是单声道，复制成双声道，方便在 Voicemeeter/EV 里看电平
    if ch == 1:
        data = np.repeat(data, 2, axis=1)
        ch = 2
        print("[PLAY] (sd) 单声道→双声道上混，确保电平条明显")

    # 播放（阻塞），指定设备
    try:
        sd.play(data, sr, device=out_device_index, blocking=True)
    except Exception as e:
        # 兜底：设备指定失败时，退回系统默认输出，至少保证能听到
        print(f"[PLAY] (sd) 设备 index={out_device_index} 打开失败：{e}，改用系统默认播放")
        sd.play(data, sr, blocking=True)

    print("[PLAY] 播放完成（sounddevice）")



# ================= 路径工具 =================
def make_out_path(wav_dir: Optional[str] = None) -> str:
    base = f"tts_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.wav"
    if wav_dir:
        os.makedirs(wav_dir, exist_ok=True)
        return os.path.join(wav_dir, base)
    tmp_dir = tempfile.gettempdir()
    return os.path.join(tmp_dir, base)


# ================= 主流程 =================
def main():
    ap = argparse.ArgumentParser("Voicemeeter 混音同传（A采英→VM混音→EV录制）")
    ap.add_argument("--mode", choices=["live","file"], default="live")
    ap.add_argument("--in_wav", type=str, default=None)
    # 采集输入与 TTS 输出
    ap.add_argument("--device-name", type=str, default="CABLE-A Output")
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--tts_device_name", type=str, default="Voicemeeter Input (VB-Audio Voicemeeter VAIO)")
    ap.add_argument("--tts_device_index", type=int, default=None)
    # 识别 & TTS
    ap.add_argument("--whisper", type=str, default="medium.en")#base  large-v3  medium.en  --whisper small.en
    ap.add_argument("--lead_ms", type=int, default=1200)
    ap.add_argument("--voice", type=str, default="zh-CN-YunxiNeural")
    # 语速/音高控制
    ap.add_argument("--tts_rate_pct", type=int, default=-20, help="基础语速百分比（-50..+50），默认-5更慢一点")
    ap.add_argument("--rate_scale", type=float, default=0.6, help="韵律驱动的语速动态缩放")
    ap.add_argument("--pitch_scale", type=float, default=1.0, help="韵律驱动的音高动态缩放")
    # 其他
    ap.add_argument("--host", type=str, default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--seg_s", type=float, default=4.0)
    ap.add_argument("--list_inputs", action="store_true")
    ap.add_argument("--keep_wav", action="store_true", help="保留每段TTS的WAV，不自动删除")
    ap.add_argument("--wav_dir", type=str, default=None, help="WAV保存目录（配合 --keep_wav）")
    args = ap.parse_args()

    if args.list_inputs:
        list_input_devices(); return

    # 输入（默认 CABLE-A Output）
    in_dev = args.device_index
    if in_dev is None:
        in_dev = find_device_index_by_name(args.device_name, want_output=False)
        if in_dev is None:
            print(f"[Err] 找不到包含“{args.device_name}”的输入设备。先 --list_inputs 查看准确名称/索引。")
            return

    # 输出（默认 Voicemeeter Input）
    out_dev_idx = args.tts_device_index
    if out_dev_idx is None:
        out_dev_idx = find_device_index_by_name(args.tts_device_name, want_output=True)
        if out_dev_idx is None:
            print(f"[Warn] 未找到输出设备包含“{args.tts_device_name}”，将使用系统默认输出（不影响功能）。")
        else:
            try:
                pa = pyaudio.PyAudio()
                info = pa.get_device_info_by_index(out_dev_idx)
                print(f"[DEV ] TTS输出设备 index={out_dev_idx} name={info.get('name')} (maxOut={info.get('maxOutputChannels')})")
            except Exception as e:
                print(f"[DEV ] TTS输出设备 index={out_dev_idx}（无法读取名称：{e}）")
            finally:
                try: pa.terminate()
                except: pass
    else:
        print(f"[DEV ] TTS输出设备 index={out_dev_idx}（来自 --tts_device_index）")

    # 初始化
    asr = ASRWhisper(args.whisper)
    mt  = MTMarian("en-zh")
    mic = None

    # 半双工 + 缓冲拼句
    mute_until = 0.0
    last_play_end = 0.0
    last_tts_text = ""
    buf_text = ""
    buf_last_add = 0.0

    print("[VM Translator] Ready. 玩法：播放器→CABLE-A Input；Voicemeeter 混音；EV 录 B1 或系统声。Ctrl+C 退出。")

    try:
        if args.mode == "live":
            mic = LineInVAD(device_index=in_dev, vad_level=2, max_segment_s=8.0, silence_ms=600)

            while True:
                now = time.time()
                if now < mute_until:
                    mic.drain(0.05); time.sleep(0.05); continue
                if (now - last_play_end) < 0.35:   # 播放后缓冲期
                    mic.drain(0.05); time.sleep(0.05); continue

                pcm = mic.read_segment()
                appended_this_round = False

                if pcm:
                    # ASR
                    txt_en, d_asr = asr.transcribe_pcm16(pcm, 16000, "en")
                    if txt_en and not is_trivial(txt_en):
                        # 追加到缓冲
                        buf_text = (buf_text + " " + txt_en).strip() if buf_text else txt_en.strip()
                        buf_last_add = time.time()
                        appended_this_round = True
                        print(f"[ASR] {txt_en}  ({d_asr:.2f}s)")
                    else:
                        print(f"[ASR]  (skip)  ({d_asr:.2f}s)")

                # 触发条件：句读 / 足够长 / 超时（仅当本轮未追加）
                hit_punct = buf_text.endswith((".", "?", "!", ";", "…", "。", "？", "！", "；"))
                hit_len   = len(buf_text) >= 120
                hit_idle  = (not appended_this_round) and (time.time() - buf_last_add > 2.8)

                if not (hit_punct or hit_len or hit_idle):
                    continue

                # 取出缓冲文本
                text_to_translate = buf_text
                buf_text = ""

                # MT + 清洗
                raw_zh, d_mt = mt.translate(text_to_translate)
                txt_zh = clean_text_for_tts(raw_zh)
                if is_trivial(txt_zh):
                    print(f"[MT ]  (skip trivial)  ({d_mt:.2f}s)")
                    continue
                # 粗略重复过滤
                if last_tts_text and len(txt_zh) > 10:
                    common = os.path.commonprefix([txt_zh, last_tts_text])
                    if len(common) / max(len(txt_zh), len(last_tts_text)) > 0.9:
                        print("[MT ] (skip duplicate-like)")
                        continue
                print(f"[MT ] {txt_zh}  ({d_mt:.2f}s)")

                # 韵律 -> rate/pitch（edge-tts 合法格式）
                if pcm:
                    y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    y = np.zeros(16000, dtype=np.float32)
                pros = prosody_from_audio(y, 16000)
                rate_str, pitch_str = map_prosody_to_params(
                    pros["syll_rate"], pros["mean_f0"],
                    base_rate_pct=args.tts_rate_pct,
                    rate_scale=args.rate_scale,
                    pitch_scale=args.pitch_scale
                )

                # 合成到 WAV（纯文本 + 合法 rate/pitch）
                out_wav = make_out_path(args.wav_dir if args.keep_wav else None)
                wav_dur = asyncio.run(edge_tts_to_wav(txt_zh, out_wav, args.voice, rate=rate_str, pitch=pitch_str))

                # 合成结果体检
                ok, dur_checked = verify_wav_ok(out_wav)
                if not ok:
                    print("[TTS] 首次合成校验失败，重试一次…")
                    try: os.remove(out_wav)
                    except Exception: pass
                    out_wav = make_out_path(args.wav_dir if args.keep_wav else None)
                    wav_dur = asyncio.run(edge_tts_to_wav(txt_zh, out_wav, args.voice, rate=rate_str, pitch=pitch_str))
                    ok, dur_checked = verify_wav_ok(out_wav)
                if not ok:
                    print("[TTS] 合成多次失败，跳过播放。")
                    try: os.remove(out_wav)
                    except Exception: pass
                    last_tts_text = txt_zh
                    continue
                wav_dur = max(wav_dur, dur_checked)

                # 同步：告知眼镜端
                # 估个 ASR/MT 开销（保守 0.6s 起）
                delay_ms = int(1000 * (max(0.6, 0.0) + max(wav_dur, 0.6)))
                tts_start_at = time.time() + args.lead_ms / 1000.0
                send_delay(delay_ms, tts_start_at, args.host, args.port)
                print(f"[SYNC] delay_ms={delay_ms}  tts_start_at={tts_start_at:.3f}  (lead={args.lead_ms}ms)")

                # 播放窗口：暂停采集 + 播放 + drain
                mute_until = tts_start_at + wav_dur + 0.60
                now2 = time.time()
                if tts_start_at > now2: time.sleep(tts_start_at - now2)

                try:
                    print(f"[PLAY] 准备播放到 out_dev_idx={out_dev_idx if out_dev_idx is not None else '默认设备'} -> {out_wav}")
                    play_wav_to_device(out_wav, out_dev_idx)
                    print("[PLAY] 播放完成")
                except Exception as e:
                    print(f"[PLAY] 播放异常：{e}")

                if not args.keep_wav:
                    try: os.remove(out_wav)
                    except Exception as e:
                        print(f"[CLEAN] 删除临时文件失败：{e}")

                last_play_end = time.time()
                mic.drain(0.70)
                last_tts_text = txt_zh
                print(f"[TTS] done wav={wav_dur:.2f}s rate={rate_str} pitch={pitch_str}")

        else:
            # 离线：处理一个wav文件
            if not args.in_wav or not os.path.exists(args.in_wav):
                print("[Err] 请提供有效 --in_wav 路径"); return
            segments = wav_segments_from_file(args.in_wav, rate=16000, seg_len=args.seg_s)
            for seg in segments:
                txt_en, d_asr = asr.transcribe_float(seg, "en")
                if not txt_en or is_trivial(txt_en):
                    print(f"[ASR]  (skip)  ({d_asr:.2f}s)"); continue
                print(f"[ASR] {txt_en}  ({d_asr:.2f}s)")
                raw_zh, d_mt = mt.translate(txt_en)          # 修复：这里用 txt_en
                txt_zh = clean_text_for_tts(raw_zh)
                if is_trivial(txt_zh):
                    print(f"[MT ]  (skip trivial)  ({d_mt:.2f}s)"); continue
                pros = prosody_from_audio(seg, 16000)
                rate_str, pitch_str = map_prosody_to_params(
                    pros["syll_rate"], pros["mean_f0"],
                    base_rate_pct=args.tts_rate_pct,
                    rate_scale=args.rate_scale,
                    pitch_scale=args.pitch_scale
                )
                out_wav = make_out_path(args.wav_dir if args.keep_wav else None)
                wav_dur = asyncio.run(edge_tts_to_wav(txt_zh, out_wav, args.voice, rate=rate_str, pitch=pitch_str))
                ok, _ = verify_wav_ok(out_wav)
                if ok:
                    play_wav_to_device(out_wav, out_dev_idx)
                else:
                    print("[TTS] 离线段合成无效，跳过。")
                if not args.keep_wav:
                    try: os.remove(out_wav)
                    except Exception: pass
                print(f"[TTS] done. wav={wav_dur:.2f}s rate={rate_str} pitch={pitch_str}")

    except KeyboardInterrupt:
        pass
    finally:
        if mic: mic.close()
        print("[VM Translator] 结束。")


if __name__ == "__main__":
    main()
