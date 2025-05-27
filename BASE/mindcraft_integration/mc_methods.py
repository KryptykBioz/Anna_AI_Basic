from datetime import datetime, timedelta, timezone  
from colorama import Fore
import socketio
import os

from personality.bot_info import botname

MIND_SERVER_HOST = os.getenv("MINDSERVER_HOST", "localhost")
MIND_SERVER_PORT = os.getenv("MINDSERVER_PORT", "8080")
AGENT_NAME = os.getenv("MINDCRAFT_AGENT_NAME", f"{botname}")

def init_mindserver(self):
    self.sio = socketio.Client(reconnection=True)
    self.sio.on("chat-message", self._on_mind_msg)
    self.sio.on("disconnect", lambda: self._init_mindserver())
    self.sio.connect(f"http://{MIND_SERVER_HOST}:{MIND_SERVER_PORT}")
    self.sio.emit("register-agents", [AGENT_NAME])
    self.sio.emit("login-agent", AGENT_NAME)
    self.mindserver_connected = True
    print(Fore.GREEN + f"[MindServer] Connected as {AGENT_NAME}" + Fore.RESET)

def on_mind_msg(self, agent, payload):
    if agent == AGENT_NAME:
        msg = payload.get("message", "")
        entry = f"[Game] {msg}"
        self.history.append({"role": "system", "content": entry})
        self.text_queue.put(entry)