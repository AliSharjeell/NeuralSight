![NeuralSight Max](NeuralSightMax.png)

# NeuralSight: Max

**Bridging the gap between human intent and digital autonomy.**

Millions of people with upper limb amputations or severe motor impairments can't access their computers as fully and freely as they want to. Right now, bridging that gap requires thousands of dollars in bulky, specialized hardware.

**We are changing that.** NeuralSight is a dual-modal accessibility suite that delivers total digital autonomy using nothing but a standard everyday webcam and a microphone.

---

## The Vision

### 1. Zero-Hardware Eyetracking
We use lightweight computer vision running locally to track your gaze. Wherever you look on the screen, your pointer follows. An intentional blink is your click. It is fluid, real-time, and completely eliminates the need for a physical mouse.

### 2. Agentic 'Computer Use'
For typing and complex tasks, we built **Max**—an autonomous, multimodal AI agent that visually interprets the screen's GUI. Give it a voice command, and the agent takes over. It navigates the OS, reasons through application states, and fills out input fields—bypassing the keyboard entirely. It can even execute intricate system tasks that the user might not know how to do themselves.

---

## Prerequisites
Before you clone and run, ensure you have the following installed:
- **Python 3.10+**: Core logic and UI.
- **Bun**: Required to run the OpenClaude gRPC server.
- **Git**: To clone the repo and its submodules.

---

## Getting Started

### 1. Clone and Install
```powershell
git clone <your-repo-url>
cd NeuralSight
pip install -r requirements.txt
```

### 2. Setup Environment
Rename `.env.example` to `.env` and fill in your keys.

### 3. Launch the Suite
The easiest way to start is using the provided batch script which handles port cleanup and dual-process orchestration:
```powershell
.\start_neuralsight.bat
```

**Manual Execution (for developers):**
If you prefer running the components separately to see detailed logs:

**Terminal 1 (Headless Server):**
```powershell
cd openclaude
.\start_openclaude_server.ps1
```

**Terminal 2 (Voice Interface):**
```powershell
python voice_terminal_pipeline.py
```

---

## Voice Interaction
- **Wake Word**: Say "Max" to activate the assistant.
- **Interruptible**: Say "Max" or "Max, stop" at any time during execution to cancel or redirect the current task.
- **Visual Feedback**: A floating, modern pill UI provides real-time waveform feedback and natural language status updates (e.g., "Thinking...", "Working on it...").

## Key Technologies
- **OpenClaude**: Headless gRPC server for agentic computer control.
- **Groq Whisper**: Ultra-low latency voice transcription (whisper-large-v3-turbo).
- **Windows-MCP**: Direct system integration for 91+ specialized tools (Files, Registry, Browser, UI).
- **CustomTkinter**: Premium, hardware-accelerated Python UI.

---

*NeuralSight: Bringing together human-centric empathy and state-of-the-art agentic workflows. No expensive rigs. Just seamless, fully autonomous computer access.*
