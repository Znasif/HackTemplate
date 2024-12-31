from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os
from dotenv import load_dotenv

@dataclass
class BotConfig:
    token: str
    command_prefix: str
    
    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            token=os.getenv('DISCORD_TOKEN'),
            command_prefix=os.getenv('COMMAND_PREFIX', '!')
        )

@dataclass
class Config:
    bot: BotConfig
    
    @classmethod
    def load(cls):
        bot_config = BotConfig.from_env()
        return cls(bot=bot_config)