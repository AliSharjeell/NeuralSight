import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import re
from pathlib import Path

# Zinc 900 Color Palette
COLOR_BG = "#09090b"  # Zinc 950
COLOR_CARD = "#18181b"  # Zinc 900
COLOR_BORDER = "#27272a"  # Zinc 800
COLOR_TEXT = "#fafafa"  # Zinc 50
COLOR_SUBTEXT = "#a1a1aa"  # Zinc 400
COLOR_ACCENT = "#3b82f6"  # Blue 500
COLOR_ACCENT_HOVER = "#2563eb"  # Blue 600
COLOR_SUCCESS = "#22c55e"  # Green 500
COLOR_WARNING = "#f59e0b"  # Amber 500

class EyeTrackerLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuralSight Head-Tracker Launcher")
        self.root.geometry("520x520")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Frame setup
        self.main_frame = tk.Frame(root, bg=COLOR_BG, padx=30, pady=30)
        self.main_frame.pack(fill="both", expand=True)

        # Header
        self.header_label = tk.Label(
            self.main_frame, 
            text="NeuralSight", 
            fg=COLOR_TEXT, 
            bg=COLOR_BG, 
            font=("Segoe UI", 24, "bold")
        )
        self.header_label.pack(pady=(0, 5))

        self.sub_header = tk.Label(
            self.main_frame, 
            text="Head-driven mouse + wink gestures", 
            fg=COLOR_SUBTEXT, 
            bg=COLOR_BG, 
            font=("Segoe UI", 10)
        )
        self.sub_header.pack(pady=(0, 20))

        # Status Card
        self.status_frame = tk.Frame(self.main_frame, bg=COLOR_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.status_frame.pack(fill="x", pady=(0, 20))
        
        self.model_path = Path("calibration_model.pkl")
        model_exists = self.model_path.exists()
        status_color = COLOR_SUCCESS if model_exists else COLOR_ACCENT
        status_text = "Model Calibrated" if model_exists else "Calibration Required"
        
        self.status_indicator = tk.Label(
            self.status_frame, 
            text=status_text, 
            fg=status_color, 
            bg=COLOR_CARD, 
            font=("Segoe UI", 9, "bold"),
            padx=15,
            pady=12
        )
        self.status_indicator.pack(side="left")

        # Controls frame
        self.controls_frame = tk.Frame(self.main_frame, bg=COLOR_BG)
        self.controls_frame.pack(fill="x", pady=(0, 15))

        self.preview_var = tk.BooleanVar(value=True)
        self.preview_check = tk.Checkbutton(
            self.controls_frame,
            text="Show webcam preview",
            variable=self.preview_var,
            fg=COLOR_SUBTEXT,
            bg=COLOR_BG,
            selectcolor=COLOR_CARD,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            font=("Segoe UI", 9),
        )
        self.preview_check.pack(anchor="w", pady=2)

        self.overlay_var = tk.BooleanVar(value=False)
        self.overlay_check = tk.Checkbutton(
            self.controls_frame,
            text="Fullscreen gaze overlay",
            variable=self.overlay_var,
            fg=COLOR_SUBTEXT,
            bg=COLOR_BG,
            selectcolor=COLOR_CARD,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            font=("Segoe UI", 9),
        )
        self.overlay_check.pack(anchor="w", pady=2)

        # Gesture hint card
        self.hint_frame = tk.Frame(self.main_frame, bg=COLOR_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.hint_frame.pack(fill="x", pady=(0, 20))
        
        hints = [
            "Left wink  → Left click",
            "Right wink → Right click",
            "Double wink→ Double click",
            "Hold left eye + move head → Drag / Scroll",
        ]
        for i, hint in enumerate(hints):
            tk.Label(
                self.hint_frame,
                text=hint,
                fg=COLOR_SUBTEXT,
                bg=COLOR_CARD,
                font=("Segoe UI", 9),
                padx=12,
                pady=3 if i == 0 else 2,
            ).pack(anchor="w")

        # Buttons
        self.create_button("QUICK START", self.run_workflow, primary=True)
        self.create_button("RE-CALIBRATE", self.run_calibration)
        self.create_button("LAUNCH TRACKER", self.run_tracker)

        # Footer
        self.footer = tk.Label(
            self.main_frame, 
            text="ESC to exit • Head Mode", 
            fg=COLOR_SUBTEXT, 
            bg=COLOR_BG, 
            font=("Segoe UI", 8)
        )
        self.footer.pack(side="bottom", pady=(15, 0))

        self.root.bind("<Escape>", lambda e: self.root.destroy())

    @staticmethod
    def _filter_stderr(stderr_text: str) -> str:
        """Remove TensorFlow/oneDNN info lines that pollute stderr."""
        if not stderr_text:
            return ""
        lines = stderr_text.splitlines()
        filtered = [
            line for line in lines
            if not re.search(r"tensorflow|oneDNN|TF_ENABLE_ONEDNN", line, re.IGNORECASE)
        ]
        return "\n".join(filtered).strip()

    def create_button(self, text, command, primary=False):
        btn_bg = COLOR_ACCENT if primary else COLOR_CARD
        btn_fg = COLOR_TEXT
        
        btn = tk.Button(
            self.main_frame,
            text=text,
            command=command,
            bg=btn_bg,
            fg=btn_fg,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            activebackground=COLOR_ACCENT_HOVER if primary else COLOR_BORDER,
            activeforeground=COLOR_TEXT,
            cursor="hand2",
            bd=0,
            pady=10
        )
        btn.pack(fill="x", pady=5)
        
        btn.bind("<Enter>", lambda e: btn.configure(bg=COLOR_ACCENT_HOVER if primary else COLOR_BORDER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=btn_bg))

    def _build_tracker_args(self):
        args = [sys.executable, "track.py"]
        if self.preview_var.get():
            args.append("--show-webcam-preview")
        if self.overlay_var.get():
            args.append("--show-overlay")
        return args

    def run_calibration(self):
        self.root.withdraw()
        try:
            result = subprocess.run([sys.executable, "calibrate.py"], capture_output=True, text=True)
            if result.returncode != 0:
                real_err = self._filter_stderr(result.stderr)
                if real_err:
                    messagebox.showerror("Calibration Error", real_err[:500])
                else:
                    messagebox.showerror("Error", "Calibration failed or was cancelled.")
            self.refresh_status()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.root.deiconify()

    def run_tracker(self):
        if not self.model_path.exists():
            messagebox.showwarning("Calibration Required", "Please calibrate the tracker first.")
            return
        
        self.root.withdraw()
        try:
            result = subprocess.run(self._build_tracker_args(), capture_output=True, text=True)
            if result.returncode != 0:
                real_err = self._filter_stderr(result.stderr)
                if real_err:
                    messagebox.showerror("Tracker Error", real_err[:500])
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.root.deiconify()

    def run_workflow(self):
        """Full flow: Calibrate -> Track"""
        self.root.withdraw()
        try:
            print("Launching Calibration...")
            result = subprocess.run([sys.executable, "calibrate.py"], capture_output=True, text=True)
            
            if result.returncode != 0:
                real_err = self._filter_stderr(result.stderr)
                if real_err:
                    messagebox.showerror("Calibration Error", real_err[:500])
                self.refresh_status()
                return
            
            if self.model_path.exists():
                print("Calibration successful. Launching Tracker...")
                subprocess.run(self._build_tracker_args())
            else:
                messagebox.showwarning("Calibration", "Calibration was cancelled or did not produce a model.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.refresh_status()
            self.root.deiconify()

    def refresh_status(self):
        model_exists = self.model_path.exists()
        status_color = COLOR_SUCCESS if model_exists else COLOR_ACCENT
        status_text = "Model Calibrated" if model_exists else "Calibration Required"
        self.status_indicator.configure(text=status_text, fg=status_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackerLauncher(root)
    root.mainloop()
