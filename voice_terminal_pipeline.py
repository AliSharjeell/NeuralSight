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
import uuid
from typing import Optional

# Force UTF-8 and High DPI awareness on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import grpc
from protos import openclaude_pb2
from protos import openclaude_pb2_grpc

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
# "max" + expanded phonetic variants for better interrupt detection
WAKE_WORD_PHRASES = ["max", "macs", "macks", "marks", "mask", "match", "mass", "mats", "maps", "makes", "mix", "marx", "necks", "next", "math", "mad", "master", "mess", "mouse"]
MICROPHONE_INDEX = None
CALIBRATION_SECONDS = 1.5
ENERGY_THRESHOLD_FLOOR = 50
PAUSE_AFTER_SPEECH = 2.0       # seconds of silence before we consider the prompt done
PHRASE_TIME_LIMIT   = 15       # max seconds of speech to capture
GRPC_HOST = "127.0.0.1"
GRPC_PORT = 50051
PROMPT_TEMPLATE = """You are Max, a high-speed Windows AI assistant. 
You are being controlled via VOICE. Keep responses extremely brief (1 sentence max).
You have FULL ACCESS to the Windows system via the windows-mcp tools.

USER COMMAND: {text}

AVAILABLE TOOLS (windows-mcp):
- App: launch/close apps (app_name="chrome", etc.)
- PowerShell: run any command
- Shortcut: keys like "ctrl+l", "ctrl+t", "enter", "win+r", "alt+tab"
- Type: type text, press_enter=true
- Click/Move/Scroll: UI interaction
- Screenshot/Snapshot: see the screen/UI tree
- FileSystem/Registry/Clipboard/Process/Wait

RULES FOR NAVIGATION & BROWSER:
1. NEVER use App(action="launch") if Chrome is already running (check Snapshot).
2. To navigate: Use Shortcut(keys="ctrl+l") to focus the current tab's address bar, then Type the URL.
3. To open a new tab: Only use Shortcut(keys="ctrl+t") if the user specifically asks for a "new tab". Otherwise, reuse the current tab.
4. NEVER type a URL into a website's internal search bar.
5. If the current tab is already the correct site (e.g., GitHub), do NOT reload it; just start the task.

HINTS:
- "X" is only for x.com (Twitter) tasks.
- If the user says "GitHub", go to github.com.
- Use direct URLs (e.g., https://google.com/search?q=...) for searches.

Execute the user's request flawlessly and quickly.
"""
DEBUG_MODE = True


# ============================================================================
# OPENCLAUDE gRPC CLIENT
# ============================================================================

class OpenClaudeGrpcClient:
    """
    Persistent gRPC client for the OpenClaude headless server.
    Uses bidirectional streaming for real-time token output and auto-approves
    tool permission requests so voice commands never hang waiting for input.
    """

    def __init__(self):
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[openclaude_pb2_grpc.AgentServiceStub] = None
        self.session_id = str(uuid.uuid4())
        self._current_call: Optional[grpc.Call] = None
        self._call_lock = threading.Lock()
        self._running = True
        self._connect()

    def _connect(self) -> None:
        target = f"{GRPC_HOST}:{GRPC_PORT}"
        for attempt in range(1, 6):
            try:
                self.channel = grpc.insecure_channel(target)
                self.stub = openclaude_pb2_grpc.AgentServiceStub(self.channel)
                grpc.channel_ready_future(self.channel).result(timeout=3)
                print(f"[gRPC] Connected to OpenClaude server at {target}")
                print(f"[gRPC] Session ID: {self.session_id[:8]}...")
                return
            except Exception as e:
                print(f"[gRPC] Connection attempt {attempt}/5 failed — {e}")
                if self.channel:
                    self.channel.close()
                self.channel = None
                self.stub = None
                if attempt < 5:
                    time.sleep(2)
        print(f"[gRPC] ERROR: Could not connect to {GRPC_HOST}:{GRPC_PORT} after 5 attempts.")
        print("[gRPC] Make sure the server is running: .\\start_neuralsight.bat")

    def send(self, text: str) -> bool:
        """
        Start a new bidirectional Chat stream with the given message.
        Cancels any in-progress stream first.  The background receive thread
        pushes tokens / tool events onto _stdout_queue in real time.

        Returns (success, done_event) where done_event is a threading.Event
        that gets set when the response stream finishes.
        """
        if not self.stub:
            print("[gRPC] Not connected. Start the server first.")
            return False

        # Cancel any in-progress stream so we don't have overlapping responses
        self.cancel()


        request_queue: queue.Queue = queue.Queue()
        request_queue.put(openclaude_pb2.ClientMessage(
            request=openclaude_pb2.ChatRequest(
                message=text,
                working_directory=os.getcwd(),
                session_id=self.session_id
            )
        ))

        def request_generator():
            while self._running:
                try:
                    msg = request_queue.get(timeout=0.1)
                    yield msg
                except queue.Empty:
                    continue

        call = self.stub.Chat(request_generator())
        with self._call_lock:
            self._current_call = call

        threading.Thread(target=self._receive, args=(call, request_queue), daemon=True).start()
        print(f"[gRPC] Sent: {text[:80]}...")
        return True, call

    def _receive(self, call: grpc.Call, request_queue: queue.Queue) -> None:
        """Background thread: iterate server messages and queue them for the UI."""
        try:
            for response in call:
                is_terminal = self._handle_response(response, request_queue)
                if is_terminal:
                    break
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.CANCELLED:
                _stdout_queue.put(f"[gRPC Error] {e.details()}")
        finally:
            with self._call_lock:
                if self._current_call is call:
                    self._current_call = None
            # Signal that this stream is done
            _done_event.set()

    def _handle_response(self, response: openclaude_pb2.ServerMessage,
                         request_queue: queue.Queue) -> bool:
        """Handle a single server response. Returns True if this was a terminal event (done/error)."""
        event = response.WhichOneof("event")
        if event == "text_chunk":
            _stdout_queue.put(response.text_chunk.text)
        elif event == "tool_start":
            _stdout_queue.put(f"[Tool: {response.tool_start.tool_name}]")
        elif event == "tool_result":
            out = response.tool_result.output
            if len(out) > 300:
                out = out[:300] + "..."
            prefix = "ERR" if response.tool_result.is_error else "OK"
            _stdout_queue.put(f"[{prefix} Result] {out}")
        elif event == "action_required":
            ar = response.action_required
            if ar.type == openclaude_pb2.ActionRequired.CONFIRM_COMMAND:
                request_queue.put(openclaude_pb2.ClientMessage(
                    input=openclaude_pb2.UserInput(
                        reply="y",
                        prompt_id=ar.prompt_id
                    )
                ))
                _stdout_queue.put(f"[Auto-approved] {ar.question}")
            else:
                _stdout_queue.put(f"[Needs input] {ar.question}")
        elif event == "done":
            _stdout_queue.put(
                f"[Done | tokens: {response.done.prompt_tokens} -> {response.done.completion_tokens}]"
            )
            return True
        elif event == "error":
            _stdout_queue.put(f"[Error] {response.error.message}")
            return True
        return False

    def cancel(self) -> None:
        """Cancel the current in-flight gRPC stream."""
        with self._call_lock:
            if self._current_call:
                try:
                    self._current_call.cancel()
                except Exception:
                    pass
                self._current_call = None

    def is_alive(self) -> bool:
        if not self.channel:
            return False
        try:
            return self.channel.get_state(try_to_connect=False) == grpc.ChannelConnectivity.READY
        except Exception:
            return False

    def terminate(self) -> None:
        self._running = False
        with self._call_lock:
            if self._current_call:
                try:
                    self._current_call.cancel()
                except Exception:
                    pass
        if self.channel:
            self.channel.close()
        print("[gRPC] Disconnected.")


# Queue visible to both threads; holds stdout lines for the UI
_stdout_queue: queue.Queue = queue.Queue()
# Event signaled when a gRPC response stream finishes (done/error)
_done_event: threading.Event = threading.Event()


# ============================================================================
# VOICE WIDGET + CONSOLE WINDOW
# ============================================================================

class NeuralSightWindow(ctk.CTk):
    """
    Compact floating pill widget — modern, translucent, centered-bottom.
    Shows state + live waveform bars. Console output goes to terminal only.
    """

    STATE_COLORS = {
        "SLEEPING":   "#18181B",   # zinc-900
        "LISTENING":  "#18181B",   # Keep consistent zinc-900
        "PROCESSING": "#18181B",
        "EXECUTING":  "#18181B",
        "ERROR":      "#18181B",
    }
    STATE_GLOW = {
        "SLEEPING":   "#3F3F46",   # zinc-700
        "LISTENING":  "#3B82F6",   # blue-500
        "PROCESSING": "#10B981",   # emerald-500
        "EXECUTING":  "#8B5CF6",   # violet-500
        "ERROR":      "#EF4444",   # red-500
    }

    WIN_W, WIN_H = 400, 52

    def __init__(self):
        super().__init__()

        # ── Window chrome ────────────────────────────────────────────────
        self.title("NeuralSight")
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        
        # Transparent corners fix for Windows
        # We set a unique color for the background and make it transparent
        TRANS_COLOR = "#000001"
        self.configure(fg_color=TRANS_COLOR)
        self.wm_attributes("-transparentcolor", TRANS_COLOR)

        # Transparency (88% opaque for the pill itself)
        self.attributes("-alpha", 0.88)

        # Center-bottom placement (Initial)
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        
        # Adjust for multiple monitors or high DPI
        # We use winfo_screenwidth() which usually refers to the primary monitor
        x = int((sw - self.WIN_W) / 2)
        y = int(sh - self.WIN_H - 160) # Moved up further per user request
        
        self.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")

        # ── Pill container ─────────────────────────────────────────────────
        self.pill = ctk.CTkFrame(
            self,
            corner_radius=26,
            border_width=0,
            fg_color=self.STATE_COLORS["SLEEPING"],
            height=self.WIN_H
        )
        self.pill.pack(fill="both", expand=True, padx=0, pady=0)
        self.pill.pack_propagate(False)

        # ── Status dot (small circle indicator) ─────────────────────────────
        self.dot_canvas = ctk.CTkCanvas(
            self.pill, width=10, height=10,
            bg=self.STATE_COLORS["SLEEPING"],
            highlightthickness=0, insertwidth=0
        )
        self.dot_canvas.pack(side="left", padx=(18, 6), pady=0)
        self._dot_id = self.dot_canvas.create_oval(1, 1, 9, 9, fill="#3F3F46", outline="")

        # ── State label ──────────────────────────────────────────────────────
        self.state_label = ctk.CTkLabel(
            self.pill, text="Say 'Max'",
            font=("Poppins", 12, "bold"), text_color="#A1A1AA"   # zinc-400
        )
        self.state_label.pack(side="left", padx=(0, 4), pady=0)

        # ── User Prompt label (italic/smaller) ──────────────────────────────────
        self.user_label = ctk.CTkLabel(
            self.pill, text="",
            font=("Poppins", 11, "italic"), text_color="#71717A"   # zinc-500
        )
        self.user_label.pack(side="left", padx=(4, 8), pady=0)

        # ── Waveform canvas (right side) ─────────────────────────────────────
        self._wave_canvas_w = 120
        self._wave_canvas_h = 36
        self.waveform = ctk.CTkCanvas(
            self.pill,
            width=self._wave_canvas_w,
            height=self._wave_canvas_h,
            bg=self.STATE_COLORS["SLEEPING"],
            highlightthickness=0, insertwidth=0
        )
        self.waveform.pack(side="right", padx=(0, 16), pady=0)

        self._wave_rms = 0.0
        self._wave_lock = threading.Lock()
        self._current_state = "SLEEPING"

        # Waveform bar history for smooth animation (High Density)
        self._num_bars = 24
        self._bar_heights = [0.0] * self._num_bars
        self._target_heights = [0.0] * self._num_bars
        self._bar_colors = ["#3B82F6", "#60A5FA", "#93C5FD", "#60A5FA", "#3B82F6"] # Blue gradient palette

        # ── Idle breathing animation ─────────────────────────────────────────
        self._breathing_phase = 0.0
        self._animate()

        # ── Keyboard + drag ──────────────────────────────────────────────────
        self.bind("<Escape>", lambda e: self._quit())
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self.pill.bind("<Button-1>", self._on_mouse_down)
        self.pill.bind("<B1-Motion>", self._on_mouse_drag)

        self.set_state("SLEEPING", "Say 'Max'")

        # Start polling stdout queue for log forwarding
        self._poll_stdout()

    # --------------------------------------------------------------------------
    # State
    # --------------------------------------------------------------------------
    def set_state(self, state: str, message: str) -> None:
        self.after(0, self._update_state, state, message)

    def _update_state(self, state: str, message: str) -> None:
        self._current_state = state
        color = self.STATE_COLORS.get(state, self.STATE_COLORS["SLEEPING"])
        glow = self.STATE_GLOW.get(state, self.STATE_GLOW["SLEEPING"])

        # Border: Glow only when EXECUTING or PROCESSING
        if state in ("EXECUTING", "PROCESSING"):
            self.pill.configure(fg_color=color, border_width=1, border_color=glow)
        else:
            self.pill.configure(fg_color=color, border_width=0)

        self.state_label.configure(text=message)
        self.waveform.configure(bg=color)
        self.dot_canvas.configure(bg=color)
        self.dot_canvas.itemconfig(self._dot_id, fill=glow)

        # Text color: bright white when active, muted when sleeping
        txt_color = "#FAFAFA" if state != "SLEEPING" else "#A1A1AA"
        self.state_label.configure(text_color=txt_color)

        # Dynamic Width: expands if text is long
        text_len = len(message) + len(self.user_label.cget("text"))
        base_w = 400
        extra = max(0, (text_len - 25) * 8)
        new_w = base_w + extra
        self.pill.configure(width=new_w)
        
        # Re-center window based on new width
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int((sw - new_w) / 2)
        y = int(sh - self.WIN_H - 160)
        self.geometry(f"{new_w}x{self.WIN_H}+{x}+{y}")

        if state not in ("LISTENING",):
            self._target_heights = [0.0] * self._num_bars

    # --------------------------------------------------------------------------
    # Waveform animation
    # --------------------------------------------------------------------------
    def update_waveform(self, rms: float) -> None:
        with self._wave_lock:
            self._wave_rms = rms

    def _animate(self) -> None:
        """30fps animation loop: smooth waveform + idle breathing."""
        import math

        state = self._current_state
        with self._wave_lock:
            rms = self._wave_rms

        cw = self._wave_canvas_w
        ch = self._wave_canvas_h

        if state == "LISTENING":
            import random
            for i in range(self._num_bars):
                center_factor = 0.3 + 0.7 * (1 - abs(i - self._num_bars // 2) / (self._num_bars // 2 + 0.01))
                jitter = random.uniform(0.7, 1.3)
                self._target_heights[i] = min(1.0, rms * center_factor * jitter * 2.5)
        elif state == "SLEEPING":
            # Zero heights when sleeping
            for i in range(self._num_bars):
                self._target_heights[i] = 0.0
        elif state in ("PROCESSING", "EXECUTING"):
            # Zero heights when working (User requested waveform only when listening)
            for i in range(self._num_bars):
                self._target_heights[i] = 0.0

        # Clear and Redraw bars with rounded caps
        self.waveform.delete("bar")
        bar_spacing = cw / self._num_bars
        for i in range(self._num_bars):
            # Smoothing
            self._bar_heights[i] += (self._target_heights[i] - self._bar_heights[i]) * 0.25
            h = self._bar_heights[i] * ch
            
            x_pos = i * bar_spacing + 2
            y_mid = ch / 2
            
            # Select color from gradient palette
            color_idx = int((i / self._num_bars) * (len(self._bar_colors) - 1))
            bar_color = self._bar_colors[color_idx]
            
            # Draw rounded bar
            self.waveform.create_line(
                x_pos, y_mid - h/2 - 1,
                x_pos, y_mid + h/2 + 1,
                width=3, fill=bar_color, capstyle="round", tags="bar"
            )

        self.after(33, self._animate)

    # --------------------------------------------------------------------------
    # Console output → show latest line in pill during EXECUTING
    # --------------------------------------------------------------------------
    def _poll_stdout(self) -> None:
        last_line = None
        try:
            while True:
                line = _stdout_queue.get_nowait()
                if line and line.strip():
                    last_line = line.strip()
        except queue.Empty:
            pass
        # Show latest output line in the pill when executing
        if last_line and self._current_state in ("EXECUTING", "PROCESSING"):
            # Truncate to fit pill width
            display = last_line[:42] + ("..." if len(last_line) > 42 else "")
            self.state_label.configure(text=display)
        self.after(50, self._poll_stdout)

    def set_user_prompt(self, text: str) -> None:
        """Update the italicized user prompt label."""
        # Truncate if too long
        display = text[:35] + ("..." if len(text) > 35 else "")
        self.after(0, lambda: self.user_label.configure(text=f"\"{display}\""))

    def log(self, text: str) -> None:
        """Forward to terminal stdout only."""
        pass

    # --------------------------------------------------------------------------
    # Drag to move
    # --------------------------------------------------------------------------
    def _on_mouse_down(self, event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_mouse_drag(self, event) -> None:
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        g = self.geometry()
        _, xs, ys = g.split("+")
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

    def __init__(self, widget: NeuralSightWindow, claude: OpenClaudeGrpcClient):
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
            # Optimize for 1-syllable wake word "Max"
            self.recognizer.pause_threshold = 0.5
            self.recognizer.phrase_threshold = 0.2
            self.recognizer.non_speaking_duration = 0.3
            
            self._print(f"[Mic] Energy threshold: {t:.1f} | Wait: {PAUSE_AFTER_SPEECH}s")
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
        # Persistent logging
        try:
            with open("neuralsight.log", "a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------------
    def _loop(self) -> None:
        self.widget.set_state("SLEEPING", "Ready for you")

        while self.running:
            try:
                self._listen_for_wake_word()
            except Exception as e:
                self._print(f"[Pipeline] Loop error: {e}")
                self.widget.set_state("ERROR", f"Error: {e}")
                time.sleep(2)

    def _listen_for_wake_word(self) -> None:
        self.widget.set_state("SLEEPING", "Say 'Max'...")

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=2)
        except sr.WaitTimeoutError:
            return
        except Exception as e:
            self._print(f"[Mic] Listen error: {e}")
            return

        # Use Groq Whisper instead of PocketSphinx — far more accurate
        text = self._quick_transcribe(audio)
        if not text:
            return

        text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()

        # Check if wake word is anywhere in the transcription
        if not self._wake_word_matches(text_clean):
            return

        self._print(f"[Wake] Heard: \"{text_clean}\"")
        self._on_wake_word_detected()

    def _quick_transcribe(self, audio_data: sr.AudioData) -> str:
        """Fast Groq Whisper transcription for wake word detection."""
        if not self._groq_client:
            # Fallback to PocketSphinx if no Groq
            try:
                return self.recognizer.recognize_sphinx(audio_data)
            except Exception:
                return ""
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            resp = self._groq_client.audio.transcriptions.create(
                file=("wake.wav", wav_bytes),
                model=GROQ_MODEL,
                temperature=0.0,
                response_format="json"
            )
            return resp.text.strip()
        except Exception:
            return ""

    def _wake_word_matches(self, text: str) -> bool:
        """Check if wake word appears anywhere in the transcribed text."""
        for phrase in WAKE_WORD_PHRASES:
            if phrase in text:
                return True
        return False

    def _on_wake_word_detected(self) -> None:
        self._print("[EVENT] Wake word detected!")
        self._beep()

        while self.running:
            time.sleep(0.1)  # let mic buffer flush
            self.widget.set_state("LISTENING", "I'm listening...")
            
            audio = self._capture_command()
            if audio is None:
                self.widget.set_state("SLEEPING", "Ready for you")
                return

            self.widget.set_state("PROCESSING", "Thinking...")
            transcribed = self._transcribe(audio)
            if not transcribed:
                self.widget.set_state("ERROR", "Try again?")
                time.sleep(1.5)
                self.widget.set_state("SLEEPING", "Say 'Max'...")
                return

            self.widget.set_user_prompt(transcribed)
            cmd = PROMPT_TEMPLATE.format(text=transcribed)
            self.widget.set_state("EXECUTING", "Working on it...")

            # Clear done event before sending so we can wait for completion
            _done_event.clear()
            result = self.claude.send(cmd)
            
            if isinstance(result, tuple):
                success, _ = result
            else:
                success = result

            if success:
                self._print(f"[CLI] Command sent. Waiting for response...")
                # Wait for completion, but check for voice interrupt
                status = self._wait_with_interrupt()
                
                if status == "NEW_COMMAND":
                    # Loop back to LISTENING immediately
                    self._print("[Interrupt] Re-listening for new command...")
                    continue
                elif status == "CANCELLED":
                    # Exit the wake loop
                    break
                else:
                    # DONE naturally
                    self.widget.set_state("SLEEPING", "Done!")
                    time.sleep(1.5)
                    break
            else:
                self._print("[CLI] Send failed.")
                self.widget.set_state("ERROR", "Connection failed")
                time.sleep(2)
                break

        self.widget.set_state("SLEEPING", "Say 'Max'...")

    def _wait_with_interrupt(self) -> str:
        """
        Wait for gRPC response, listening for 'Josh' to interrupt.
        Returns:
            "DONE": finished naturally
            "CANCELLED": user said 'Josh stop'
            "NEW_COMMAND": user said 'Josh' (new command follows)
        """
        while not _done_event.is_set():
            if _done_event.wait(timeout=0.4):
                return "DONE"

            # Quick listen for interrupt wake word
            try:
                with self.microphone as source:
                    # Faster listen for interrupt
                    audio = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=1.2)
                text = self._quick_transcribe(audio)
                if not text:
                    continue
                text_lower = text.lower().strip()

                if self._wake_word_matches(text_lower):
                    self._print(f"[Interrupt] Heard: \"{text_lower}\"")
                    self._beep(freq=800, dur=80)

                    # Cancel current execution
                    self.claude.cancel()
                    _done_event.set()
                    self._print("[Interrupt] Cancelled current task.")

                    # Check if it's "stop" or "cancel"
                    if any(word in text_lower for word in ["stop", "cancel", "shut up", "quit"]):
                        self._print("[Interrupt] Stopped.")
                        return "CANCELLED"

                    # Otherwise, user wants to say a new command
                    return "NEW_COMMAND"
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self._print(f"[Interrupt] Error: {e}")
                continue
        return "DONE"
    # --------------------------------------------------------------------------
    # Command capture with live RMS waveform
    # --------------------------------------------------------------------------
    def _capture_command(self):
        """Listen with live RMS waveform animation; stop on silence or phrase limit."""
        self._print("[Recorder] Listening...")
        try:
            with self.microphone as source:
                # Start a background thread to feed RMS to waveform
                self._recording = True
                rms_thread = threading.Thread(
                    target=self._rms_monitor, args=(source,), daemon=True
                )
                rms_thread.start()

                audio = self.recognizer.listen(
                    source,
                    timeout=None,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )
                self._recording = False
            self._print(f"[Recorder] Captured.")
            return audio
        except sr.WaitTimeoutError:
            self._recording = False
            self._print("[Recorder] No speech detected.")
            return None
        except Exception as e:
            self._recording = False
            self._print(f"[Recorder] Error: {e}")
            return None

    def _rms_monitor(self, source) -> None:
        """Feed live RMS to the waveform while recording."""
        import pyaudio
        stream = source.stream
        while self._recording:
            try:
                # Read a small chunk for RMS calculation
                if hasattr(stream, 'stream') and stream.stream and stream.stream.is_active():
                    data = stream.stream.read(1024, exception_on_overflow=False)
                    rms = rms_from_bytes(data)
                    self.widget.update_waveform(rms)
                else:
                    self.widget.update_waveform(0.15)  # gentle pulse if can't read
            except Exception:
                self.widget.update_waveform(0.15)
            time.sleep(0.05)  # 20fps RMS updates

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

    # ── OpenClaude gRPC client ────────────────────────────────────────────────
    claude = OpenClaudeGrpcClient()

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