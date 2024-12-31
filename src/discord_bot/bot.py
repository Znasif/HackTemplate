from discord.ext import commands
import discord
from config import Config
from .commands import CommandRegistry, PingCommand

class AIChatBot(commands.Bot):
    def __init__(self, config: Config, intents: discord.Intents = discord.Intents.default()):
        super().__init__(command_prefix=config.bot.command_prefix, intents=intents)
        self._config = config
        self.command_registry = CommandRegistry()
        self._setup_commands()
    
    def _setup_commands(self):
        # Register built-in commands
        self.command_registry.register('ping', PingCommand())
        
        # Create Discord.py command wrappers
        for cmd_name, cmd_instance in self.command_registry._commands.items():
            @self.command(name=cmd_name)
            async def command_wrapper(ctx, cmd=cmd_instance):
                await cmd.execute(ctx)
    
    async def setup_hook(self):
        print(f'Logged in as {self.user}')