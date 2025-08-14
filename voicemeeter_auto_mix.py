import time
import comtypes.client
import os

def vs_init():
    """
    通过 COM 接口初始化 Voicemeeter API
    """
    try:
        vm = comtypes.client.CreateObject("VoicemeeterRemote.VBVMR")
        return vm
    except Exception as e:
        print("初始化 Voicemeeter COM 接口失败：", e)
        return None

def main():
    vm = vs_init()
    if not vm:
        return

    print("初始化 Voicemeeter 控制...")

    # 连接到当前 Voicemeeter 实例（Banana 或 Potato）
    result = vm.Login()
    if result != 0:
        print("无法登录 Voicemeeter 控制接口，返回码：", result)
        return
    print("已登录 Voicemeeter")

    # 稍等让状态稳定
    time.sleep(0.5)

    # 设置硬件输入 1 = CABLE-A Output（同传中文）
    vm.SetParameter("Strip[0].HardwareInput.Device", "CABLE-A Output (VB-Audio Virtual Cable)")

    # 设置硬件输入 2 = System Default or Mic for English (原声)
    vm.SetParameter("Strip[1].HardwareInput.Device", "Microphone (Realtek High Definition Audio)")

    # 路由设置：要同时送到 A1（监听）与 B1（录音虚拟输出）
    vm.SetParameter("Strip[0].A1", 1)
    vm.SetParameter("Strip[0].B1", 1)
    vm.SetParameter("Strip[1].A1", 1)
    vm.SetParameter("Strip[1].B1", 1)

    # 设置 A1 为你的扬声器；B1 为 VB-Audio Cable A Input（用于 EV 录音）
    vm.SetParameter("Bus[0].HardwareOutput.Device", "Speakers (Realtek High Definition Audio)")       # A1
    vm.SetParameter("Bus[1].HardwareOutput.Device", "CABLE-A Input (VB-Audio Virtual Cable)")          # B1

    vm.Flush()
    print("Voicemeeter 路由配置完成！")

    # 若要脚本持续保持运行，可以睡眠或退出根据需求
    # time.sleep(2)
    vm.Logout()

if __name__ == "__main__":
    main()

