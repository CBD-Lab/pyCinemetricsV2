import sys
import os
import platform
import importlib
import contextlib


def resource_path(relative_path):
    myos = platform.system()

    if (myos == 'Windows'):
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    elif (myos == 'Darwin') or (myos == 'Linux'):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    return None


class Splash:
    def __init__(self):
        if '_PYIBoot_SPLASH' in os.environ and \
                importlib.util.find_spec('pyi_splash'):
            import pyi_splash  # type: ignore
            self.splash = pyi_splash
        else:
            self.splash = None

    def close(self):
        if self.splash:
            self.splash.close()

    def update(self, text):
        if self.splash:
            self.splash.update_text(text)


@contextlib.contextmanager
def suppress_c_stderr():
    """抑制 C 库（如 FFmpeg）直接输出到 stderr 的错误/警告信息。
    
    用于抑制类似 "[rv40 @ ...] Invalid decoder state: B-frame without reference data" 这类消息。
    通过重定向底层文件描述符实现（不影响 Python 的 logging/stderr）。
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)  # 复制当前 stderr 文件描述符
    os.dup2(devnull, 2)     # 将 stderr 重定向到 null
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)  # 恢复原始 stderr
        os.close(devnull)
        os.close(old_stderr)
