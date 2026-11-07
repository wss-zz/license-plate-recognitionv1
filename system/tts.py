# -*- coding: utf-8 -*-
"""语音播报：Windows SAPI（离线，无需联网/第三方包）。"""
import queue
import threading

_q = queue.Queue()
_started = False
_lk = threading.Lock()


def _worker():
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()      # 子线程必须初始化 COM，否则 Dispatch 失败
        sp = win32com.client.Dispatch("SAPI.SpVoice")
        sp.Rate = 2          # -10..10
        sp.Volume = 100
    except Exception as e:  # noqa: BLE001
        print("[tts] SAPI 不可用:", e)
        sp = None
    while True:
        text = _q.get()
        if text is None:
            break
        if sp is None:
            continue
        try:
            sp.Speak(text)
        except Exception as e:  # noqa: BLE001
            print("[tts] speak failed:", e)


def speak(text, block=False):
    """默认异步播报，避免阻塞请求；block=True 用于演示脚本。"""
    global _started
    with _lk:
        if not _started:
            threading.Thread(target=_worker, daemon=True).start()
            _started = True
    if block:
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            win32com.client.Dispatch("SAPI.SpVoice").Speak(text)
            return True
        except Exception:  # noqa: BLE001
            return False
    _q.put(text)
    return True


if __name__ == "__main__":
    speak("系统启动，请通行", block=True)
