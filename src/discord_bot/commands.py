from abc import ABC, abstractmethod
from typing import Dict, Any
from discord.ext import commands

class BaseCommand(ABC):
    @abstractmethod
    async def execute(self, ctx: commands.Context, *args, **kwargs) -> None:
        pass

class PingCommand(BaseCommand):
    async def execute(self, ctx: commands.Context, *args, **kwargs) -> None:
        await ctx.send('Pong!')

class CommandRegistry:
    def __init__(self):
        self._commands: Dict[str, BaseCommand] = {}
    
    def register(self, name: str, command: BaseCommand) -> None:
        self._commands[name] = command
    
    def get_command(self, name: str) -> BaseCommand:
        return self._commands.get(name)