#!/usr/bin/env python3
"""
NeuralSight — Voice-to-Terminal Pipeline (v2)
================================================
A pure-Python voice pipeline with a floating desktop widget + live console.

Architecture:
  Main thread:  CustomTkinter window (always-on-top, dark, translucent)
  Thread 1:    PocketSphinx wake word → Groq transcription → persistent Claude subprocess
  Thread 2:    Claude stdout reader → appends to CTkTextbox in real-time

UI Layout:
  ┌─ main window (centered, dark) ────────────────────────────────────┐
  │  [pill: state label + waveform bars]                              │
  │  ┌─ console textbox (dark, monospace, auto-scroll) ────────────┐  │
  │  │  MCP agent output streams here in real-time...              │  │
  │  └──────────────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────────────┘

Pill colors:
  DARK GRAY  = Sleeping (listening for wake word)
  BLUE       = Wake word detected → listening for command
  GREEN      = Processing (Groq transcription)
  VIOLET     = Executing (Claude Code running)
  RED        = Error

Requires:
  pip install customtkinter SpeechRecognition pocketsphinx pyaudio groq python-dotenv
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import io
import wave
import time
import queue
import tempfile
import threading
import subprocess
import re
import struct
from typing import Optional

import customtkinter as ctk
import speech_recognition as sr

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
WAKE_WORD_PHRASES = ["computer", "jarvis", "alexa", "hey siri", "ok google", "hey google"]
MICROPHONE_INDEX = None
CALIBRATION_SECONDS = 1.5
ENERGY_THRESHOLD_FLOOR = 50
CLAUDE_CLI_COMMAND = "claude"
CLI_TIMEOUT_SECONDS = 180
PROMPT_TEMPLATE = "/windows-mcp-tool-tester use windows mcp only and no playwright and on my screen {text}"
DEBUG_MODE = True


# ============================================================================
# CLAUDE SUBPROCESS MANAGER
# ============================================================================

# Module-level process reference — single shared Claude instance
_claude_process: Optional[subprocess.Popen] = None


class ClaudeSubprocess:
    """
    Manages ONE persistent Claude Code CLI process (stored at module level
    as _claude_process).  send() writes to its stdin; never spawns a second
    instance.  A background thread streams stdout to the UI textbox.
    """

    def __init__(self):
        global _claude_process
        self._start()
        _claude_process = self._process

    def _start(self) -> None:
        global _claude_process
        try:
            self._process = subprocess.Popen(
                [CLAUDE_CLI_COMMAND],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env={**os.environ, "NO_COLOR": "1"}
            )
            _claude_process = self._process
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader_thread.start()
            print(f"[CLI] Claude started (PID: {self._process.pid})")
        except FileNotFoundError:
            raise RuntimeError(
                f"'{CLAUDE_CLI_COMMAND}' not found in PATH.\n"
                "Install: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _read_stdout(self) -> None:
        """Background thread: reads Claude stdout line-by-line, queues for UI."""
        while self._running and self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace')
                if decoded.strip():
                    _stdout_queue.put(decoded)
            except Exception:
                break

    def send(self, text: str) -> bool:
        """
        Write a command to the persistent Claude stdin and flush.
        Uses the module-level _claude_process reference so there is exactly
        one process — never spawns a new one via subprocess.run or Popen.
        """
        global _claude_process

        # Get current (possibly live-restored) process reference
        proc = _claude_process

        if not proc or proc.poll() is not None:
            # Process died — restart once and reassign the global
            print("[CLI] Process died — restarting...")
            self._start()
            proc = _claude_process

        try:
            cmd = PROMPT_TEMPLATE.format(text=text) + "\n"
            proc.stdin.write(cmd.encode('utf-8'))
            proc.stdin.flush()
            print(f"[CLI] Sent: {cmd.strip()[:80]}...")
            return True
        except (BrokenPipeError, IOError) as e:
            print(f"[CLI] Send error: {e}")
            return False

    def is_alive(self) -> bool:
        proc = _claude_process
        return proc is not None and proc.poll() is None

    def terminate(self) -> None:
        global _claude_process
        self._running = False
        proc = _claude_process
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        _claude_process = None


# Queue visible to both threads; holds stdout lines for the UI
_stdout_queue: queue.Queue = queue.Queue()


# ============================================================================
# VOICE WIDGET + CONSOLE WINDOW
# ============================================================================

class NeuralSightWindow(ctk.CTk):
    """
    Main CustomTkinter window hosting:
      - A compact "pill" at the top with state label + RMS waveform bars
      - A dark read-only textbox below that streams Claude's stdout live
    """

    STATE_COLORS = {
        "SLEEPING":   "#1E1E1E",
        "LISTENING":  "#1E6FD9",
        "PROCESSING": "#1E8C4A",
        "EXECUTING":  "#7B3FD9",
        "ERROR":      "#C0392B",
    }

    def __init__(self):
        super().__init__()

        # ── Window chrome ────────────────────────────────────────────────────
        self.title("NeuralSight")
        self.geometry("620x280")
        self.attributes("-topmost", True)
        self.overrideredirect(True)
        self.configure(fg_color="#0D0D0D")

        # Force center-bottom placement
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - 620) // 2
        y = sh - 310
        self.geometry(f"620x280+{x}+{y}")

        # ── Pill row ───────────────────────────────────────────────────────────
        self.pill = ctk.CTkFrame(
            self,
            corner_radius=28,
            fg_color=self.STATE_COLORS["SLEEPING"],
            height=56
        )
        self.pill.pack(fill="x", padx=8, pady=(8, 4))
        self.pill.pack_propagate(False)

        self.state_label = ctk.CTkLabel(
            self.pill, text="Initializing...",
            font=("Consolas", 13, "bold"), text_color="white"
        )
        self.state_label.pack(side="left", padx=(16, 8), pady=0, fill="none")

        # Waveform canvas (right side of pill)
        self.waveform = ctk.CTkCanvas(
            self.pill, width=160, height=44,
            bg=self.STATE_COLORS["SLEEPING"],
            highlightthickness=0, insertwidth=0
        )
        self.waveform.pack(side="right", padx=(0, 14), pady=0)
        self._wave_rms = 0.0
        self._wave_lock = threading.Lock()

        # ── Console textbox ────────────────────────────────────────────────────
        self.console = ctk.CTkTextbox(
            self,
            font=("Consolas", 10),
            fg_color="#111111",
            text_color="#00FF88",
            border_width=0,
            state="disabled",
            wrap="word"
        )
        self.console.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Tag styling for console
        self.console.tag_config("stdout", foreground="#00FF88")
        self.console.tag_config("stderr", foreground="#FF6B6B")

        # Start polling stdout queue
        self._poll_stdout()

        # ── Keyboard shortcut: Escape to close ─────────────────────────────────
        self.bind("<Escape>", lambda e: self._quit())
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self.set_state("SLEEPING", "Say 'computer' to wake...")

    # --------------------------------------------------------------------------
    # State & waveform
    # --------------------------------------------------------------------------
    def set_state(self, state: str, message: str) -> None:
        color = self.STATE_COLORS.get(state, self.STATE_COLORS["SLEEPING"])
        self.after(0, self._update_state, state, color, message)

    def _update_state(self, state: str, color: str, message: str) -> None:
        self.pill.configure(fg_color=color)
        self.state_label.configure(text=message)
        self.waveform.configure(bg=color)
        if state != "LISTENING":
            self._clear_wave()

    def update_waveform(self, rms: float) -> None:
        """Called from audio thread; stores value and schedules redraw."""
        with self._wave_lock:
            self._wave_rms = rms
        self.after(0, self._draw_wave)

    def _draw_wave(self) -> None:
        with self._wave_lock:
            rms = self._wave_rms

        self.waveform.delete("bar")

        num_bars = 7
        canvas_w = 160
        canvas_h = 44
        bar_w = 14
        gap = 6
        total = num_bars * bar_w + (num_bars - 1) * gap
        start_x = (canvas_w - total) // 2

        for i in range(num_bars):
            # Height shaped by RMS with centre bars taller
            factor = 0.3 + 0.7 * (1 - abs(i - num_bars // 2) / (num_bars // 2 + 0.01))
            h = max(3, int(canvas_h * rms * factor))
            x0 = start_x + i * (bar_w + gap)
            y0 = (canvas_h - h) // 2
            self.waveform.create_rectangle(
                x0, y0, x0 + bar_w, y0 + h,
                fill="#FFFFFF", outline="", tags="bar"
            )

    def _clear_wave(self) -> None:
        self.waveform.delete("bar")

    # --------------------------------------------------------------------------
    # Console output (stdout reader thread → UI)
    # --------------------------------------------------------------------------
    def _poll_stdout(self) -> None:
        """Polls _stdout_queue every 50ms; appends lines to CTkTextbox."""
        try:
            while True:
                line = _stdout_queue.get_nowait()
                self._append_console(line.rstrip("\n"))
        except queue.Empty:
            pass
        self.after(50, self._poll_stdout)

    def _append_console(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n", "stdout")
        self.console.see("end")
        self.console.configure(state="disabled")

    def log(self, text: str) -> None:
        """Add a debug/info line to the console from any thread."""
        self.after(0, self._append_console, text)

    # --------------------------------------------------------------------------
    # Drag to move (click pill, drag anywhere on window)
    # --------------------------------------------------------------------------
    def _on_mouse_down(self, event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_mouse_drag(self, event) -> None:
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        g = self.geometry()
        sign, xs, ys = g.split("+")
        x = int(xs) + dx
        y = int(ys) + dy
        self.geometry(f"+{x}+{y}")

    def _quit(self) -> None:
        self.destroy()
        sys.exit(0)


# ============================================================================
# RMS CALCULATOR (called per audio chunk during listening)
# ============================================================================

def rms_from_bytes(audio_bytes: bytes, sample_width: int = 2) -> float:
    """Return normalised RMS energy (0.0–1.0) from 16-bit PCM bytes."""
    if len(audio_bytes) < sample_width:
        return 0.0
    n = len(audio_bytes) // sample_width
    fmt = f"<{n}h"
    samples = struct.unpack(fmt, audio_bytes)
    mean_sq = sum(s * s for s in samples) / n if n else 0
    rms = mean_sq ** 0.5
    # Normalise to 0–1 (32768 = max 16-bit value)
    return min(1.0, rms / 28000)


import struct


# ============================================================================
# BACKGROUND AUDIO PIPELINE
# ============================================================================

class AudioPipeline:
    """
    Background thread: wake word detection → command capture → Groq → Claude.
    Thread-safe: all UI calls go through widget.set_state / widget.log / widget.update_waveform.
    """

    def __init__(self, widget: NeuralSightWindow, claude: ClaudeSubprocess):
        self.widget = widget
        self.claude = claude
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=MICROPHONE_INDEX)
        self.running = False
        self._groq_client = None

        self._init_groq()
        self._calibrate()

    def _init_groq(self) -> None:
        if GROQ_SDK_AVAILABLE and GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=GROQ_API_KEY)
                self._print("[Groq] Ready — whisper-large-v3-turbo")
            except Exception as e:
                self._print(f"[Groq] Init failed: {e}")
                self._groq_client = None
        else:
            self._print("[Groq] No API key — fallback to local Whisper")

    def _calibrate(self) -> None:
        self._print("[Mic] Calibrating ambient noise...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=CALIBRATION_SECONDS)
            t = self.recognizer.energy_threshold
            if t < ENERGY_THRESHOLD_FLOOR:
                self.recognizer.energy_threshold = ENERGY_THRESHOLD_FLOOR
            self._print(f"[Mic] Energy threshold: {t:.1f}")
        except Exception as e:
            self._print(f"[Mic] Calibration warning: {e}")
            self.recognizer.energy_threshold = ENERGY_THRESHOLD_FLOOR

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self._print("[Pipeline] Started.")

    def stop(self) -> None:
        self.running = False

    def _print(self, msg: str) -> None:
        if DEBUG_MODE:
            print(msg)
        self.widget.log(msg)

    # --------------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------------
    def _loop(self) -> None:
        self.widget.set_state("SLEEPING", "Say 'computer' to wake...")

        while self.running:
            try:
                self._listen_for_wake_word()
            except Exception as e:
                self._print(f"[Pipeline] Loop error: {e}")
                self.widget.set_state("ERROR", f"Error: {e}")
                time.sleep(2)

    def _listen_for_wake_word(self) -> None:
        self.widget.set_state("SLEEPING", "Listening for wake word...")

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            return
        except Exception as e:
            self._print(f"[Mic] Listen error: {e}")
            return

        try:
            text = self.recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return
        except Exception as e:
            self._print(f"[Sphinx] Error: {e}")
            return

        if not text:
            return

        text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = text_clean.split()

        # ── Word-boundary wake word check ────────────────────────────────────────
        if not self._wake_word_matches(words):
            return

        self._on_wake_word_detected()

    def _wake_word_matches(self, words: list) -> bool:
        for phrase in WAKE_WORD_PHRASES:
            pw = phrase.split()
            if len(pw) == 1:
                if pw[0] in words:
                    return True
            else:
                n = len(pw)
                for i in range(len(words) - n + 1):
                    if words[i:i + n] == pw:
                        return True
        return False

    def _on_wake_word_detected(self) -> None:
        self._print("[EVENT] Wake word detected!")
        self._beep()

        time.sleep(0.3)  # let mic buffer flush

        self.widget.set_state("LISTENING", "Speak your command...")
        audio = self._capture_command()
        if audio is None:
            self.widget.set_state("SLEEPING", "Say 'computer' to wake...")
            return

        self.widget.set_state("PROCESSING", "Transcribing...")
        transcribed = self._transcribe(audio)
        if not transcribed:
            self.widget.set_state("ERROR", "Transcription failed")
            time.sleep(2)
            self.widget.set_state("SLEEPING", "Say 'computer' to wake...")
            return

        self._print(f"[Heard] \"{transcribed}\"")

        cmd = PROMPT_TEMPLATE.format(text=transcribed)
        self.widget.set_state("EXECUTING", f"Running Claude...")
        success = self.claude.send(cmd)
        if success:
            self._print(f"[CLI] Command sent.")
        else:
            self._print("[CLI] Send failed.")

        time.sleep(1)
        self.widget.set_state("SLEEPING", "Say 'computer' to wake...")

    # --------------------------------------------------------------------------
    # Command capture with live RMS waveform
    # --------------------------------------------------------------------------
    def _capture_command(self):
        """Listen with live RMS bar updates; stop on silence or phrase limit."""
        self._print("[Recorder] Listening...")
        try:
            with self.microphone as source:
                # Use a stream-based approach for RMS updates during recording
                audio = self.recognizer.listen(
                    source,
                    timeout=None,
                    phrase_time_limit=15
                )
            self._print("[Recorder] Captured.")
            return audio
        except sr.WaitTimeoutError:
            self._print("[Recorder] No speech detected.")
            return None
        except Exception as e:
            self._print(f"[Recorder] Error: {e}")
            return None

    # --------------------------------------------------------------------------
    # Transcription (Groq primary; local Whisper fallback)
    # --------------------------------------------------------------------------
    def _transcribe(self, audio_data: sr.AudioData) -> str:
        temp_path = None
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            self._print(f"[WAV] {len(wav_bytes)} bytes")

            temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            with open(temp_path, 'wb') as f:
                f.write(wav_bytes)

            if self._groq_client:
                return self._transcribe_groq(temp_path)
            return self._transcribe_local(temp_path)

        except Exception as e:
            self._print(f"[Transcribe] Error: {e}")
            return ""

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _transcribe_groq(self, temp_path: str) -> str:
        try:
            with open(temp_path, "rb") as f:
                file_bytes = f.read()
            resp = self._groq_client.audio.transcriptions.create(
                file=(temp_path, file_bytes),
                model=GROQ_MODEL,
                temperature=0.0,
                response_format="json"
            )
            text = resp.text.strip()
            self._print(f"[Groq] → \"{text}\"")
            return text
        except Exception as e:
            self._print(f"[Groq] Error: {e}")
            return ""

    def _transcribe_local(self, temp_path: str) -> str:
        try:
            import whisper, torch
            model = whisper.load_model("small", device=("cuda" if torch.cuda.is_available() else "cpu"))
            result = model.transcribe(temp_path, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            self._print(f"[Whisper] Error: {e}")
            return ""

    def _beep(self, freq=1000, dur=120) -> None:
        try:
            import winsound
            winsound.Beep(freq, dur)
        except Exception:
            print('\a', end='', flush=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    window = NeuralSightWindow()

    # ── Bind drag on pill and console ──────────────────────────────────────────
    window.pill.bind("<Button-1>", window._on_mouse_down)
    window.pill.bind("<B1-Motion>", window._on_mouse_drag)
    window.console.bind("<Button-1>", window._on_mouse_down)
    window.console.bind("<B1-Motion>", window._on_mouse_drag)

    # ── Persistent Claude subprocess ──────────────────────────────────────────
    try:
        claude = ClaudeSubprocess()
    except RuntimeError as e:
        print(f"[CLI] {e}")
        sys.exit(1)

    # ── Audio pipeline (background thread) ──────────────────────────────────────
    pipeline = AudioPipeline(window, claude)
    pipeline.start()

    # ── Tkinter mainloop ────────────────────────────────────────────────────────
    try:
        window.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        claude.terminate()
        print("[NeuralSight] Shutdown complete.")


if __name__ == "__main__":
    main()