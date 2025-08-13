# 延时眼镜与实时翻译耳机同步系统

*DelaySync_Translator_Demo: Engineering Demonstration*

---

## 📖 项目简介

本项目为 **《延时眼镜与实时翻译耳机同步系统》** 的可运行工程演示程序，旨在展示专利技术的 **可实施性** 与 **工程实现路径**。
系统模拟了 **翻译耳机** 与 **延时眼镜** 之间的延迟匹配与嘴型画面同步过程，通过软件实现原型验证，为未来硬件实现（AR眼镜+智能耳机）奠定基础。

---

## 🏗 工程特点

- **完整双端架构**：包含“耳机端”与“眼镜端”两大模块，支持独立运行与网络通信
- **实时延迟匹配**：耳机端计算翻译延迟并发送至眼镜端，眼镜端实现视频缓冲与同步播放
- **多源视频支持**：支持 HTTP/MJPEG 与 RTSP/H.264 视频流
- **动态延迟调节**：运行中可通过 `↑/↓` 实时调整延迟（步进 100ms）
- **可移植性强**：可部署至树莓派、NVIDIA Jetson、AR眼镜等嵌入式平台
- **低延迟视频处理**：基于 OpenCV 实现高效帧缓存与延时播放

---

## 🔍 系统架构
````markdown
## 🔍 系统架构

```text
+----------------+       UDP/JSON       +------------------+
|  翻译耳机模拟   | <------------------->|   延时眼镜模拟    |
| Earbud Sim     |                      | Glasses Sim      |
|                |                      |                  |
| - 翻译延迟计算  |                      | - 视频缓冲管理    |
| - TTS播放      |                      | - 延迟显示控制    |
| - 延迟参数发送  |                      | - 环境自适应调节  |
+----------------+                      +------------------+
       ↑                                          ↑
       |                                          |
   麦克风采音                                  手机摄像头推流
````


---

## 📸 演示截图

**1️⃣ 系统运行画面**
![demo_screenshot](screenshots/run_demo.jpg)

**2️⃣ 动态调节延迟**
![delay_adjust](screenshots/delay_adjust.jpg)

**3️⃣ 手机端视频推流界面**
![phone_stream](screenshots/phone_stream_ui.jpg)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 手机端配置

* 安装 **IP Webcam**（安卓）或 **Iriun Webcam**（iOS/安卓）
* 确保手机与电脑在同一局域网
* 启动推流并记下视频流地址：

  * HTTP/MJPEG: `http://<手机IP>:8080/video`
  * RTSP/H.264: `rtsp://<手机IP>:<端口>/<路径>`

### 3. 启动延时眼镜端

```bash
python glasses_sim.py --video "http://<手机IP>:8080/video"
```

### 4. 启动翻译耳机端

```bash
python earbud_sim.py --delay_ms 1200 --text "你好，这是一次同步演示。"
```

---

## 🖥 运行效果

* 眼镜端启动后进入 **BUFFERING** 状态，等待视频缓冲
* 耳机端发送延迟参数并播放翻译语音
* 眼镜端切换至 **PLAY** 模式，延时显示嘴型画面，实现音视频同步
* 按 `↑/↓` 动态调整延迟，观察同步效果变化
* 按 `q` 退出

---

## 📂 项目结构

```
DelaySync_Translator_Demo/
│── earbud_sim.py           # 翻译耳机模拟端
│── glasses_sim.py          # 延时眼镜模拟端
│── common.py               # 公共配置
│── requirements.txt        # 依赖清单
│── README.md               # 项目说明
│── /screenshots/           # 演示截图
│── /docs/                  # 附加文档（技术交底书、流程图等）
```

---

🛠 应用价值

* 国际会议同声翻译眼镜
* 跨国商务交流辅助
* 智能旅游语音导览
* 影视现场配音同步工具

架构可扩展至多语种翻译、AR字幕叠加、3D嘴型合成等高级功能，具有商业化潜力。

---

## 🤝 合作与交流

欢迎学术机构、硬件厂商、AR/VR开发团队合作，共同推进技术落地。

---

**© 2025 DelaySync\_Translator\_Demo | Engineering Prototype**
