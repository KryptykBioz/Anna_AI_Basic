# Anna AI - Advanced Ollama Chat Bot

An intelligent AI assistant with voice interaction, vision capabilities, web search, and persistent memory powered by Ollama.

## Features

- 🗣️ **Voice Interaction**: Real-time speech-to-text and text-to-speech
- 👁️ **Vision Capabilities**: Screenshot analysis and visual understanding
- 🧠 **Memory System**: Persistent conversation memory with embeddings
- 🔍 **Web Search**: Integrated web search functionality
- 🎮 **Gaming Integration**: Minecraft server integration via MindCraft
- 📚 **Training Mode**: Supervised learning with response approval
- 🎯 **Multiple Models**: Supports different Ollama models for text and vision

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Compatible Ollama models (see Configuration section)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Anna_AI
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Vosk Speech Recognition Model

Download and extract the Vosk model:

```bash
# Create models directory
mkdir -p models

# Download Vosk model (small English model ~40MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

# Extract the model
unzip vosk-model-small-en-us-0.15.zip -d models/

# Rename for easier access (optional)
mv models/vosk-model-small-en-us-0.15 models/vosk-model-small-en-us
```

**Alternative manual download:**
1. Visit [Vosk Models](https://alphacephei.com/vosk/models)
2. Download `vosk-model-small-en-us-0.15.zip`
3. Extract to `models/vosk-model-small-en-us-0.15/`

### 5. Install Audio Dependencies

#### Windows:
```bash
# Install additional audio libraries
pip install sounddevice vosk pyaudio
```

#### macOS:
```bash
# Install PortAudio first
brew install portaudio
pip install sounddevice vosk pyaudio
```

#### Linux (Ubuntu/Debian):
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
pip install sounddevice vosk pyaudio
```

### 6. Install and Configure Ollama

#### Install Ollama:
```bash
# Visit https://ollama.ai/ for installation instructions
# Or use quick install:
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Pull Required Models:
```bash
# Text generation model (choose one)
ollama pull llama3.2:3b        # Lightweight option
ollama pull llama3.1:8b        # Balanced option
ollama pull llama3.1:70b       # High-quality option

# Vision model
ollama pull llama3.2-vision:11b
# or
ollama pull llava:13b

# Embedding model
ollama pull nomic-embed-text
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_ENDPOINT=http://localhost:11434
TEXT_LLM_MODEL=llama3.2:3b
VISION_LLM_MODEL=llama3.2-vision:11b
EMBED_MODEL=nomic-embed-text

# Memory Configuration
MAX_CONTEXT_TOKENS=500

# MindCraft Integration (optional)
MINDSERVER_HOST=localhost
MINDSERVER_PORT=8080
MINDCRAFT_AGENT_NAME=Anna
```

### Bot Configuration

Edit `personality/bot_info.py` to customize your bot:

```python
botname = "Anna"           # Bot's name
username = "User"          # Your name
textmodel = "llama3.2:3b"  # Default text model
visionmodel = "llama3.2-vision:11b"  # Default vision model
embedmodel = "nomic-embed-text"      # Embedding model
```

### System Prompt

Customize the bot's personality in `personality/SYS_MSG.py`.

## Usage

### Starting the Bot

```bash
# Activate virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Start the bot
python bot.py
```

### Interaction Modes

When you start the bot, you'll be prompted to choose a mode:

#### 1. **Text Mode** (`text`)
- Pure text-based interaction
- Type messages and receive text responses
- Best for general conversation

```
Mode (text/talk/vision/game/minecraft/train): text
```

#### 2. **Talk Mode** (`talk`)
- Voice interaction with speech-to-text and text-to-speech
- Speak naturally to the bot
- Responses are spoken back through VB-Cable

#### 3. **Vision Mode** (`vision`)
- Includes screenshot analysis capabilities
- Use keywords like "screen", "see", "look", "monitor"
- Bot can describe what's on your screen

#### 4. **Game Mode** (`game`)
- Optimized for gaming scenarios
- Includes vision capabilities for game analysis
- Faster response times

#### 5. **Minecraft Mode** (`minecraft`)
- Integrates with MindCraft server
- Allows bot interaction within Minecraft
- Requires MindCraft server setup

#### 6. **Training Mode** (`train`)
- Supervised learning mode
- Review and approve bot responses before saving
- Useful for improving bot behavior

### Built-in Commands

During any conversation, you can use these commands:

- `/memory` - Display memory statistics and recent conversations
- `/summarize` - Generate weekly summaries of past conversation history
- `exit` - Quit the application

### Example Interactions

#### Text Mode:
```
User: What's the weather like today?
Anna: I don't have access to real-time weather data, but I can help you find weather information if you'd like to search for it.

User: search current weather in New York  
Anna: [Performs web search and provides weather information]
```

#### Vision Mode:
```
User: What do you see on my screen?
Anna: [Takes screenshot and analyzes it]
I can see you have a code editor open with a Python file. There's a file explorer on the left showing various Python modules and configuration files...
```

## Project Structure

```
Anna_AI/
├── bot.py                      # Main bot script
├── BASE/                       # Core functionality
│   ├── tools/                  # Utility tools
│   │   ├── text_to_voice.py   # TTS functionality
│   │   ├── voice_to_text.py   # STT functionality  
│   │   └── query.py           # Web search tools
│   ├── memory_methods/         # Memory management
│   │   └── summarizer.py      # Conversation summarization
│   └── mindcraft_integration/  # Minecraft integration
├── personality/                # Bot configuration
│   ├── bot_info.py            # Bot identity settings
│   ├── SYS_MSG.py             # System prompt
│   └── memory/                # Persistent memory storage
│       ├── memory.json        # Conversation history
│       └── embeddings.json    # Vector embeddings
├── models/                     # Vosk models directory
└── requirements.txt           # Python dependencies
```

## Memory System

The bot maintains persistent memory through:

- **Conversation History**: Stored in `personality/memory/memory.json`
- **Vector Embeddings**: Semantic search in `personality/memory/embeddings.json`
- **Context Retrieval**: Relevant memories are retrieved for each conversation

## Troubleshooting

### Common Issues:

#### 1. **Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service if needed
ollama serve
```

#### 2. **Vosk Model Not Found**
```bash
# Verify model path
ls models/vosk-model-small-en-us-0.15/

# Update path in voice_to_text.py if necessary
```

#### 3. **Audio Device Issues**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Update device selection in voice_to_text.py
```

#### 4. **Memory/Storage Issues**
```bash
# Clear memory files if corrupted
rm personality/memory/memory.json
rm personality/memory/embeddings.json
```

### Performance Optimization:

1. **Model Selection**: Use smaller models for faster responses
2. **Context Limit**: Adjust `MAX_CONTEXT_TOKENS` for memory usage
3. **Vision Processing**: Disable vision mode if not needed
4. **Embedding Cache**: Keep embeddings.json for faster semantic search

## Advanced Features

### Custom Tool Integration

Add custom tools by implementing functions in the `BASE/tools/` directory and importing them in `bot.py`.

### Multi-Modal Responses

The bot auAnnatically switches between text and vision models based on user input keywords and context.

### Memory Management

- User-initiated weekly conversation summarization
- Semantic search through conversation history
- Configurable memory retention limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

None

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama documentation
3. Open an issue on the repository

---

**Note**: This bot requires significant computational resources for vision and large language models. Ensure your system meets the requirements for smooth operation.