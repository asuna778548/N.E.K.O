# N-03 overlay window probe: transparent + click-through + rect-follow, proven programmatically.
#
# What this proves (no Live2D, empty frame — the minimal N-03 deliverable):
#   1. An always-on-top layered window with near-zero alpha renders invisibly.
#   2. WS_EX_TRANSPARENT makes the OS hit-test skip it: WindowFromPoint over the
#      overlay rect returns the *target* window underneath, not the overlay.
#   3. A follow loop keeps the overlay rect glued to the target window rect.
#
# Run with the repo venv python: .venv/Scripts/python.exe scripts/n03_overlay_window_probe.py
# Prints a JSON verdict. Creates its own throwaway target window (tkinter) so it
# never touches the real Godot client; the follow/hit-test logic is the same
# code path N-03 will use against the consciousness window.
import ctypes
import ctypes.wintypes as wt
import json
import sys
import threading
import time
import tkinter as tk

user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
SWP_NOACTIVATE = 0x0010
SWP_ASYNCWINDOWPOS = 0x4000  # overlay window has no message pump on its owner thread;
                             # the synchronous cross-thread SetWindowPos would block forever
SWP_MOVE_FLAGS = SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS
LWA_ALPHA = 0x2
HWND_TOPMOST = wt.HWND(-1)


def find_target_hwnd(title: str) -> int:
    result = ctypes.c_void_p()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.GetWindowTextLengthW(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value == title:
                result.value = hwnd
                return False
        return True

    user32.EnumWindows(cb, 0)
    return result.value or 0


def rect_of(hwnd: int):
    r = wt.RECT()
    user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def main() -> int:
    title = "PROBE_TARGET_WIN_N03"
    target = tk.Tk()
    target.title(title)
    target.geometry("320x240+80+80")
    target.configure(bg="navy")
    tk.Label(target, text="hit-test target", fg="white", bg="navy").pack(expand=True)
    target.update()

    thwnd = find_target_hwnd(title)
    if not thwnd:
        print(json.dumps({"verdict": "FAIL", "reason": "target window not found"}))
        return 1

    hinst = ctypes.windll.kernel32.GetModuleHandleW(None)
    wc = ctypes.create_unicode_buffer("N03OverlayProbe")
    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", ctypes.c_void_p),
                    ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
                    ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                    ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p)]
    user32.RegisterClassW(ctypes.byref(WNDCLASS(style=0, lpfnWndProc=ctypes.cast(user32.DefWindowProcW, ctypes.c_void_p),
                                               hInstance=hinst, lpszClassName=wc.value)))
    x, y, w, h = rect_of(thwnd)
    ohwnd = user32.CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
        wc.value, "n03_overlay", WS_POPUP | WS_VISIBLE,
        x, y, w, h, None, None, hinst, None)
    if not ohwnd:
        print(json.dumps({"verdict": "FAIL", "reason": f"CreateWindowExW err={ctypes.GetLastError()}"}))
        return 1
    user32.SetLayeredWindowAttributes(ctypes.c_void_p(ohwnd), 0, 3, LWA_ALPHA)  # near-invisible

    stop = threading.Event()

    def follow():
        try:
            while not stop.is_set():
                fx, fy, fw, fh = rect_of(thwnd)
                user32.SetWindowPos(ctypes.c_void_p(ohwnd), HWND_TOPMOST,
                                    fx, fy, fw, fh, SWP_MOVE_FLAGS)
                time.sleep(0.05)
        except Exception as exc:  # daemon threads die silently otherwise
            print(f"DBG follow thread died: {exc!r}", file=sys.stderr)

    ft = threading.Thread(target=follow, daemon=True)
    ft.start()
    time.sleep(0.3)

    exstyle = user32.GetWindowLongW(ctypes.c_void_p(ohwnd), GWL_EXSTYLE)
    style_ok = bool(exstyle & WS_EX_TRANSPARENT) and bool(exstyle & WS_EX_LAYERED)

    ox, oy, ow, oh = rect_of(ohwnd)
    tx, ty, tw, th = rect_of(thwnd)
    aligned = (ox, oy, ow, oh) == (tx, ty, tw, th)

    GA_ROOT = 2
    pt = wt.POINT(tx + tw // 2, ty + th // 2)
    hit = user32.WindowFromPoint(pt)
    hit_root = user32.GetAncestor(ctypes.c_void_p(hit), GA_ROOT) if hit else 0
    hit_through = (hit_root == thwnd)  # hit-test lands inside the target (any child), skipping the overlay
    if hit != thwnd:
        buf = ctypes.create_unicode_buffer(64)
        buf_cls = ctypes.create_unicode_buffer(64)
        if hit:
            user32.GetWindowTextW(ctypes.c_void_p(hit), buf, 64)
            user32.GetClassNameW(ctypes.c_void_p(hit), buf_cls, 64)
        print(f"DBG hit={hit} class={buf_cls.value!r} title={buf.value!r} target={thwnd} overlay={ohwnd}",
              file=sys.stderr)

    tx2p, ty2p, tw2p, th2p = rect_of(thwnd)
    r2 = user32.SetWindowPos(ctypes.c_void_p(ohwnd), HWND_TOPMOST, tx2p, ty2p, tw2p, th2p, SWP_MOVE_FLAGS)
    print(f"DBG manual SetWindowPos ret={r2} err={ctypes.GetLastError()}", file=sys.stderr)

    target.geometry("400x300+200+160")
    target.update()
    time.sleep(0.35)
    ox2, oy2, ow2, oh2 = rect_of(ohwnd)
    tx2, ty2, tw2, th2 = rect_of(thwnd)
    follow_ok = (ox2, oy2, ow2, oh2) == (tx2, ty2, tw2, th2)

    verdict = {
        "verdict": "PASS" if (style_ok and aligned and hit_through and follow_ok) else "FAIL",
        "style_transparent_layered": style_ok,
        "rect_aligned_initial": aligned,
        "hit_test_passes_through": hit_through,
        "follow_after_move": follow_ok,
        "overlay_rect_after_move": [ox2, oy2, ow2, oh2],
        "target_rect_after_move": [tx2, ty2, tw2, th2],
    }
    stop.set()
    user32.DestroyWindow(ctypes.c_void_p(ohwnd))
    target.destroy()
    print(json.dumps(verdict))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
