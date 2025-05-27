import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Add BASE directory and root directory to path
import sys
from pathlib import Path
import asyncio

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent  # Toma_AI/BASE
ROOT_DIR = BASE_DIR.parent     # Toma_AI
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Import the local bot and system prompt
import bot
from personality.SYS_MSG import system_prompt
from personality.bot_info import botColor, userColor, systemColor

class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VTuberAI Chat")
        self.root.configure(bg="#2e2e2e")  # dark background

        # Create menu bar
        self.create_menu()

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            bg="#1e1e1e", fg="#ffffff", insertbackground="#ffffff",
            font=("Consolas", 12)
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Entry field
        entry_frame = tk.Frame(self.root, bg="#2e2e2e")
        entry_frame.pack(fill=tk.X, padx=10, pady=(0,10))

        self.user_entry = tk.Entry(
            entry_frame, bg="#3e3e3e", fg="#ffffff", insertbackground="#ffffff",
            font=("Consolas", 12)
        )
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.user_entry.bind('<Return>', self.send_message)

        send_button = tk.Button(
            entry_frame, text="Send", command=self.send_message,
            bg="#007acc", fg="#ffffff", activebackground="#005f99",
            font=("Consolas", 12), relief=tk.FLAT
        )
        send_button.pack(side=tk.RIGHT)

        # Status frame
        status_frame = tk.Frame(self.root, bg="#2e2e2e")
        status_frame.pack(fill=tk.X, padx=10, pady=(0,5))

        self.status_label = tk.Label(
            status_frame, text="Ready", bg="#2e2e2e", fg="#ffffff",
            font=("Consolas", 10)
        )
        self.status_label.pack(side=tk.LEFT)

        # Configure tags for colored messages
        self.chat_display.tag_config("user", foreground=userColor)
        self.chat_display.tag_config("bot", foreground=botColor)
        self.chat_display.tag_config("system", foreground=systemColor)
        self.chat_display.tag_config("info", foreground="#ffaa00")

        # Initialize bot and event loop
        self.bot = bot.VTuberAI()
        self.loop = bot.async_loop
        
        # Display system prompt
        self.display_message(system_prompt.strip(), tag="system")
        self.display_message("=== GUI Chat Started ===", tag="info")
        
        # Show memory stats
        self.show_memory_stats()

    def create_menu(self):
        """Create menu bar with memory options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Memory menu
        memory_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Memory", menu=memory_menu)
        memory_menu.add_command(label="Show Memory Stats", command=self.show_memory_stats)
        memory_menu.add_command(label="Show Recent Memory", command=self.show_recent_memory)
        memory_menu.add_command(label="Show Embeddings", command=self.show_embeddings)
        memory_menu.add_separator()
        memory_menu.add_command(label="Summarize Memory", command=self.summarize_memory)

    def display_message(self, message, tag="bot"):
        """Display message in chat with specified tag/color"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + '\n', tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)

    def send_message(self, event=None):
        """Send user message and get bot response"""
        user_text = self.user_entry.get().strip()
        if not user_text:
            return
        
        self.user_entry.delete(0, tk.END)
        self.display_message(f"You: {user_text}", tag="user")
        self.update_status("Processing...")

        # Run bot response in thread to avoid blocking UI
        threading.Thread(target=self.get_bot_response, args=(user_text,), daemon=True).start()

    def get_bot_response(self, text):
        """Get bot response using the full processing pipeline (includes memory saving)"""
        try:
            # Use _process_prompt_async which saves to memory (same as terminal mode)
            coro = self.bot._process_prompt_async(text, mode="text")
            # Submit coroutine to existing running loop
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            reply = future.result(timeout=120)
            
            # Display the response
            self.display_message(f"{self.bot.botname}: {reply}", tag="bot")
            self.update_status("Ready")
            
        except Exception as e:
            reply = f"[Error] {e}"
            self.display_message(f"{self.bot.botname}: {reply}", tag="bot")
            self.update_status("Error occurred")

    def show_memory_stats(self):
        """Show memory statistics"""
        memory_count = len(self.bot.memory)
        embeddings_count = len(self.bot.embeddings_data)
        stats_msg = f"=== MEMORY STATS ===\nTotal entries: {memory_count}\nTotal embeddings: {embeddings_count}"
        self.display_message(stats_msg, tag="info")

    def show_recent_memory(self):
        """Show recent memory entries"""
        if not self.bot.memory:
            self.display_message("No memory entries found.", tag="info")
            return
        
        self.display_message("=== RECENT MEMORY ===", tag="info")
        recent = self.bot.memory[-10:] if len(self.bot.memory) > 10 else self.bot.memory
        
        for entry in recent:
            role = entry.get('role', '')
            content = entry.get('content', '')
            timestamp = entry.get('timestamp', '')
            
            # Truncate long content for display
            display_content = content[:100] + "..." if len(content) > 100 else content
            
            if role == 'user':
                self.display_message(f"[{timestamp}] {self.bot.username}: {display_content}", tag="user")
            elif role == 'assistant':
                self.display_message(f"[{timestamp}] {self.bot.botname}: {display_content}", tag="bot")

    def show_embeddings(self):
        """Show embedded summaries"""
        if not self.bot.embeddings_data:
            self.display_message("No embeddings found.", tag="info")
            return
        
        self.display_message("=== EMBEDDED SUMMARIES ===", tag="info")
        recent_embeddings = self.bot.embeddings_data[-5:]  # Show last 5 summaries
        
        for emb in recent_embeddings:
            text = emb.get('text', '')
            timestamp = emb.get('timestamp', '')
            
            # Truncate long text for display
            display_text = text[:100] + "..." if len(text) > 100 else text
            self.display_message(f"[{timestamp}] {display_text}", tag="system")

    def summarize_memory(self):
        """Trigger memory summarization"""
        self.display_message("=== STARTING MEMORY SUMMARIZATION ===", tag="info")
        self.update_status("Summarizing memory...")
        
        # Run summarization in thread to avoid blocking UI
        def run_summarize():
            try:
                self.bot.summarize_memory()
                self.display_message("=== MEMORY SUMMARIZATION COMPLETE ===", tag="info")
                self.update_status("Ready")
            except Exception as e:
                self.display_message(f"[Error during summarization] {e}", tag="info")
                self.update_status("Summarization error")
        
        threading.Thread(target=run_summarize, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatGUI(root)
    root.mainloop()