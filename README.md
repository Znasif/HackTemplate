# Discord AI Assistant

A modular Discord bot that processes audio/video streams and static content to provide AI-powered assistance. The bot captures desktop video using PyAutoGUI, processes audio inputs, and generates responses using LLM technology.

## Features

- **Discord Integration**: Basic command system with extensible architecture
- **Screen Capture**: Desktop video capture using PyAutoGUI
- **Audio Processing**: Voice command recognition and text-to-speech responses
- **LLM Integration**: AI-powered responses using LLAMA 3
- **Static Content**: Process PDFs and webpages for context
- **Modular Design**: Easy to extend and customize

## Prerequisites

- Python 3.10+
- Conda
- WSL2 (for Linux-based deployment)
- Discord Bot Token
- X Server (if running on WSL2)

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Znasif/discord-ai-assistant.git
cd discord-ai-assistant
```

2. **Create and activate conda environment**
```bash
conda create -n discord-ai-env python=3.10
conda activate discord-ai-env
```

3. **Install dependencies**
```bash
conda install -c conda-forge discord.py python-dotenv pytest pyyaml pyautogui pillow numpy
```

4. **Configure environment**
Create a `.env` file in the root directory:
```env
DISCORD_TOKEN=your_discord_token_here
COMMAND_PREFIX=!
```

5. **Run the bot**
```bash
python src/main.py
```

## Project Structure

```
discord-ai-assistant/
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── discord_bot/
│   │   ├── bot.py
│   │   └── commands.py
│   ├── input_handlers/
│   ├── static_content/
│   ├── audio_processing/
│   └── video_processing/
└── tests/
```

## Extending the Bot

### Adding New Commands

1. Create a new command class:
```python
from discord_bot.commands import BaseCommand

class YourCommand(BaseCommand):
    async def execute(self, ctx, *args, **kwargs):
        # Your command logic here
        pass
```

2. Register the command:
```python
bot.command_registry.register('your_command', YourCommand())
```

### Adding Video Processing

The video processing module uses PyAutoGUI for screen capture:

```python
from video_processing import DesktopVideoProcessor

processor = DesktopVideoProcessor()
frame = processor.capture_frame()
```

### Custom Configuration

Add new configuration sections by extending the Config class:

```python
@dataclass
class YourConfig:
    # Your config options
    pass

@dataclass
class Config:
    bot: BotConfig
    your_config: YourConfig
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## WSL2 Setup

If running on WSL2, ensure X Server is configured:

1. Install VcXsrv on Windows
2. Set DISPLAY variable in WSL2:
```bash
export DISPLAY=:0
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Write tests for new features
4. Submit a pull request

## Architecture

The bot follows a modular architecture with several key components:

1. **Command System**: Extensible command handling
2. **Video Processing**: Screen capture and analysis
3. **Audio Processing**: Voice input/output
4. **LLM Integration**: AI response generation
5. **Static Content**: Document processing

Each component is designed to be independently extensible through abstract base classes.

## Running in Production

For production deployment:

1. Set up proper logging
```python
from utils.logging import setup_logging
setup_logging(level='INFO')
```

2. Configure error handling
```python
from utils.error_handler import ErrorHandler
bot.add_error_handler(ErrorHandler())
```

3. Use a process manager (e.g., PM2 or supervisord)

## Limitations

- Screen capture is system-dependent
- LLM responses may require significant resources
- Audio processing requires proper hardware setup

## License

[MIT]

## Acknowledgments

- Discord.py developers
- PyAutoGUI team
- LLAMA framework contributors