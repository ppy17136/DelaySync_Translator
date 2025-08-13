# glasses_sim.py
import cv2, time, json, socket, threading, argparse
from collections import deque
from common import HOST, PORT, VIDEO_SRC

class DelayController:
    def __init__(self, default_delay_ms=1000):
        self.delay_ms = default_delay_ms
        self.tts_start_at = None
        self.lock = threading.Lock()

    def update(self, delay_ms: int, tts_start_at: float | None):
        with self.lock:
            self.delay_ms = max(0, int(delay_ms))
            self.tts_start_at = tts_start_at

    def get(self):
        with self.lock:
            return self.delay_ms, self.tts_start_at

def udp_listener(ctrl: DelayController, host: str, port: int, on_msg=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[Glasses] UDP 监听 {host}:{port}")
    while True:
        data, _ = sock.recvfrom(2048)
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            continue
        if msg.get("type") == "delay_update":
            delay_ms = int(msg.get("delay_ms", 1000))
            tts_start_at = msg.get("tts_start_at", None)
            ctrl.update(delay_ms, tts_start_at)
            if on_msg:
                on_msg(delay_ms, tts_start_at)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=VIDEO_SRC, help="手机相机的网络流地址（HTTP或RTSP）")
    ap.add_argument("--host", type=str, default=HOST, help="UDP监听地址（与earbud_sim一致）")
    ap.add_argument("--port", type=int, default=PORT, help="UDP监听端口（与earbud_sim一致）")
    ap.add_argument("--default_delay_ms", type=int, default=1000, help="默认延迟（未收到消息前）")
    args = ap.parse_args()

    ctrl = DelayController(args.default_delay_ms)

    def on_msg(delay_ms, tts_start_at):
        print(f"[Glasses] 收到延迟: {delay_ms} ms, tts_start_at={tts_start_at}")

    t = threading.Thread(target=udp_listener, args=(ctrl, args.host, args.port, on_msg), daemon=True)
    t.start()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Glasses] 无法打开视频源：{args.video}")
        return

    # 用一个双端队列缓存帧，每个元素是 (timestamp, frame)
    buffer = deque()
    playback_started = False
    last_info_time = 0.0
    fps_est = 0.0
    fps_alpha = 0.1
    last_frame_time = None

    print("[Glasses] 开始拉流。按 ↑/↓ 动态调延迟 100ms，按 q 退出。")

    while True:
        ok, frame = cap.read()
        now = time.time()
        if not ok or frame is None:
            # 轻微等待避免空转
            time.sleep(0.01)
            continue

        # 粗略估计输入fps（指数平滑）
        if last_frame_time is not None:
            inst_fps = 1.0 / max(1e-6, (now - last_frame_time))
            fps_est = fps_alpha * inst_fps + (1 - fps_alpha) * fps_est
        else:
            fps_est = 0.0
        last_frame_time = now

        # 加入缓冲
        buffer.append((now, frame))

        # 控制缓冲与播放逻辑
        delay_ms, tts_start_at = ctrl.get()
        delay_s = delay_ms / 1000.0

        # 当 earbud 告知一个 tts_start_at 时，我们希望画面在 (tts_start_at + delay_s) 左右开始“与TTS对齐显示”
        target_play_start = None
        if tts_start_at is not None:
            target_play_start = tts_start_at  # 我们的策略：画面的时间轴延后 delay_s 显示，所以此处先把播放启动点对齐到 tts_start_at
                                              # 具体显示的帧选择为“当前时刻 - delay_s”的对应帧

        # 缓冲是否足够：以最新帧时间戳 - 最老帧时间戳 来估计缓存时长
        buffer_span = 0.0
        if len(buffer) >= 2:
            buffer_span = buffer[-1][0] - buffer[0][0]

        # 规则：
        # 1) 未开始播放前：如果收到 tts_start_at，则等到 now >= tts_start_at 且 buffer_span >= delay_s 开始播放；
        #    未收到 tts_start_at，就等到 buffer_span >= delay_s 再开始。
        # 2) 播放时：总是选择“当前时刻 - delay_s”最接近的帧进行显示（即固定延迟显示）
        if not playback_started:
            if target_play_start is not None:
                if (now >= target_play_start) and (buffer_span >= delay_s):
                    playback_started = True
            else:
                if buffer_span >= delay_s:
                    playback_started = True

        display_frame = None
        if playback_started:
            # 目标显示时间戳（希望显示 delay_s 之前的画面）
            target_ts = now - delay_s

            # 从缓冲中找到时间戳最接近 target_ts 的帧（以及把更早的帧丢弃）
            while len(buffer) >= 2 and buffer[1][0] <= target_ts:
                buffer.popleft()
            # 现在 buffer[0] 是最接近或稍大的帧
            display_frame = buffer[0][1].copy()
        else:
            # 尚未开始播放，展示最新帧并叠加“缓冲中”提示
            display_frame = buffer[-1][1].copy()

        # 叠加屏显信息
        info_lines = []
        info_lines.append(f"Delay: {delay_ms} ms | Buffer: {buffer_span:.2f}s | FPS_in: {fps_est:.1f}")
        info_lines.append(f"Mode: {'PLAY' if playback_started else 'BUFFERING'} | q退出  Up/Down調延遲")
        y = 30
        for line in info_lines:
            cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
            y += 28

        cv2.imshow("Glasses (Delayed Mouth View)", display_frame)

        # 按键交互：↑/↓ 改延迟，q 退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 82:   # Up arrow
            d, tts = ctrl.get()
            ctrl.update(d + 100, tts)
        elif key == 84:   # Down arrow
            d, tts = ctrl.get()
            ctrl.update(max(0, d - 100), tts)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
