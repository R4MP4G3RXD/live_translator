#!/usr/bin/env python3
"""
Live Translator – light client
==============================
Microphone  ->  your Live Translator server (recognition + translation)  ->  black window + narration.

Needs only:  pip install numpy sounddevice edge-tts pygame     (see requirements-client.txt)
Keep this file next to live_translator.py (it uses it as a library; nothing heavy is loaded).

Run:  python translator_client.py
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import live_translator as lt

CONFIG_PATH = Path(__file__).resolve().parent / "client_config.json"
FIELDS = ("net_host", "net_port", "net_token", "net_tls", "net_tls_ca", "input_device", "source_lang",
          "target_lang", "narrate", "tts_voice", "fullscreen_on_start", "save_review_files",
          "review_dir", "log_dir", "mute_input_while_speaking", "continuous_mode")


class ClientApp:
    def __init__(self, root: tk.Tk, cfg_path: Path = CONFIG_PATH):
        self.root, self.cfg_path = root, cfg_path
        self.cfg = lt.load_config(cfg_path)
        self.cfg["net_mode"] = "client"
        self.session = None
        self.v: dict[str, tk.Variable] = {}
        root.title("Live Translator – client")

        bar = ttk.Frame(root)
        bar.pack(side="bottom", fill="x", padx=10, pady=(4, 10))
        f = ttk.Frame(root, padding=(12, 10))
        f.pack(side="top", fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        try:
            devices = lt.list_input_devices()
        except Exception:
            devices = []
        dev_labels = [lt.DEFAULT_DEV_LABEL] + [l for _, l in devices]
        cur = self.cfg["input_device"] if self.cfg["input_device"] in dev_labels else lt.DEFAULT_DEV_LABEL

        rows = [
            ("net_host", "Server address", "text", None),
            ("net_port", "Port", "text", None),
            ("net_token", "Access token", "password", None),
            ("input_device", "Microphone / line-in", "combo", dev_labels),
            ("source_lang", "Spoken language", "combo", lt.SRC_NAMES),
            ("target_lang", "Translate to", "combo", lt.TGT_NAMES),
        ]
        r = 0
        for key, label, kind, values in rows:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=4)
            if key == "input_device":
                var = tk.StringVar(value=cur)
            elif key == "source_lang":
                var = tk.StringVar(value=lt.Launcher._name_of(self.cfg[key], True))
            elif key == "target_lang":
                var = tk.StringVar(value=lt.Launcher._name_of(self.cfg[key], False))
            else:
                var = tk.StringVar(value=str(self.cfg[key]))
            if kind == "combo":
                w = ttk.Combobox(f, textvariable=var, values=values, state="readonly", width=42)
            else:
                w = ttk.Entry(f, textvariable=var, width=44, show="•" if kind == "password" else "")
            w.grid(row=r, column=1, sticky="ew", padx=4, pady=4)
            self.v[key] = var
            r += 1
        for key, label in (("narrate", "Narrate the translation"),
                           ("net_tls", "Encrypted connection (TLS)"),
                           ("fullscreen_on_start", "Start full screen"),
                           ("save_review_files", "Save transcript + translation files for review")):
            var = tk.BooleanVar(value=bool(self.cfg[key]))
            ttk.Checkbutton(f, text=label, variable=var).grid(row=r, column=0, columnspan=2,
                                                              sticky="w", padx=4, pady=2)
            self.v[key] = var
            r += 1
        ttk.Label(f, text="Voice: automatic for the language. Window keys: F11 full screen, "
                          "Esc leave it, + / − text size.", foreground="#666", wraplength=420
                  ).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0))

        self.start_btn = ttk.Button(bar, text="▶  Start", command=self.on_start)
        self.start_btn.pack(side="right", ipadx=16, ipady=3)
        ttk.Button(bar, text="Test connection", command=self.on_test).pack(side="right", padx=8)
        self.msg = ttk.Label(bar, text="", foreground="#666")
        self.msg.pack(side="left")
        root.protocol("WM_DELETE_WINDOW", self.on_quit)
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(max(520, root.winfo_reqwidth()), int(sw * 0.95))
        h = min(max(380, root.winfo_reqheight()), int(sh * 0.92))
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 4)}")

    # ---- settings ----
    def collect(self) -> dict:
        cfg = dict(self.cfg)
        for k, var in self.v.items():
            val = var.get()
            if k == "source_lang":
                cfg[k] = lt.SRC_BY_NAME.get(val, "auto")
            elif k == "target_lang":
                cfg[k] = lt.TGT_BY_NAME.get(val, "sr-Latn")
            elif k == "input_device":
                cfg[k] = "" if val == lt.DEFAULT_DEV_LABEL else val
            elif k == "net_port":
                try:
                    cfg[k] = int(str(val).strip())
                except ValueError:
                    cfg[k] = 0
            elif isinstance(val, bool) or isinstance(var, tk.BooleanVar):
                cfg[k] = bool(val)
            else:
                cfg[k] = str(val).strip()
        host = cfg["net_host"]
        if host.count(":") == 1 and host.rsplit(":", 1)[1].isdigit():  # "pc:9000" is accepted too
            cfg["net_host"], cfg["net_port"] = host.rsplit(":", 1)[0], int(host.rsplit(":", 1)[1])
        cfg["net_mode"] = "client"
        return cfg

    def save(self, cfg: dict) -> None:
        self.cfg = cfg
        try:
            lt.save_config(self.cfg_path, {k: cfg[k] for k in lt.DEFAULTS})
        except OSError as e:
            messagebox.showwarning("Could not save settings", str(e))

    # ---- actions ----
    def on_test(self):
        cfg = self.collect()
        err = lt.check_config(cfg)
        if err:
            messagebox.showwarning("Check settings", err)
            return
        self.msg.configure(text="Connecting…")
        self.root.update_idletasks()
        probe = lt.ClientSession(cfg, self.root, lambda: None)
        try:
            conn = probe._connect()
            conn.close()
            self.msg.configure(text="Connection works.")
        except Exception as e:
            self.msg.configure(text="")
            messagebox.showerror("Connection failed", str(e))

    def on_start(self):
        cfg = self.collect()
        err = lt.check_config(cfg)
        if err:
            messagebox.showwarning("Check settings", err)
            return
        self.save(cfg)
        self.start_btn.state(["disabled"])
        self.msg.configure(text="Running… close the black window to stop.")
        self.session = lt.ClientSession(cfg, self.root, self.on_closed)
        self.session.start()

    def on_closed(self):
        self.session = None
        self.start_btn.state(["!disabled"])
        self.msg.configure(text="")

    def on_quit(self):
        if self.session:
            self.session.request_close()
        try:
            self.save(self.collect())
        except Exception:
            pass
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    ClientApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
