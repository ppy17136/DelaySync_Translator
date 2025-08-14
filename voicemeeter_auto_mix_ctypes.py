# voicemeeter_auto_mix_ctypes.py
import os, time, ctypes
from ctypes import wintypes

# 常见安装路径（可按需补充）
CANDIDATES = [
    r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll",
    r"C:\Program Files\VB\Voicemeeter\VoicemeeterRemote64.dll",
    r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote.dll",  # 32位备用
]

def load_vm_dll():
    for p in CANDIDATES:
        if os.path.exists(p):
            return ctypes.WinDLL(p)
    raise FileNotFoundError("未找到 VoicemeeterRemote DLL，请确认已安装 Voicemeeter。")

def set_param_str(dll, name: str, value: str):
    # int VBVMR_SetParameterStringW(LPCWSTR szParamName, LPCWSTR szString)
    fn = dll.VBVMR_SetParameterStringW
    fn.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    fn.restype  = ctypes.c_int
    rc = fn(name, value)
    if rc != 0:
        print(f"[Warn] SetParameterStringW 失败 {name}={value} rc={rc}")

def set_param_float(dll, name: str, val: float):
    # int VBVMR_SetParameterFloat(LPCSTR szParamName, float fValue) - 注意有的版本是 A/W 两套
    try:
        fn = dll.VBVMR_SetParameterFloat
        fn.argtypes = [wintypes.LPCSTR, ctypes.c_float]
        fn.restype  = ctypes.c_int
        rc = fn(name.encode("ascii"), ctypes.c_float(val))
    except AttributeError:
        # 某些版本只有 W 宽字符接口
        fn = dll.VBVMR_SetParameterFloatW
        fn.argtypes = [wintypes.LPCWSTR, ctypes.c_float]
        fn.restype  = ctypes.c_int
        rc = fn(name, ctypes.c_float(val))
    if rc != 0:
        print(f"[Warn] SetParameterFloat 失败 {name}={val} rc={rc}")

def main():
    dll = load_vm_dll()

    # 登录
    login = dll.VBVMR_Login
    login.restype = ctypes.c_long
    if login() != 0:
        raise RuntimeError("VBVMR_Login 失败，请确保 Voicemeeter 已运行（并以管理员安装过）。")

    time.sleep(0.2)

    # 打开路由：Strip[0]/[1] → A1 + B1
    for i in (0, 1):
        set_param_float(dll, f"Strip[{i}].A1", 1.0)
        set_param_float(dll, f"Strip[{i}].B1", 1.0)

    # 可选：把推子置于 0 dB
    for i in (0, 1):
        set_param_float(dll, f"Strip[{i}].Gain", 0.0)

    # 注：设备选择（A1/B1 具体绑定哪个声卡）建议在 GUI 中设一次，Voicemeeter 会保存。
    # 如需脚本切设备，可查官方 PDF（Remote API）相应字段；不同版本命名可能不同，这里不强行写死。

    time.sleep(0.2)
    logout = dll.VBVMR_Logout
    logout.restype = ctypes.c_long
    logout()
    print("Voicemeeter 路由配置完成（ctypes）。")

if __name__ == "__main__":
    main()
