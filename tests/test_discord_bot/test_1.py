import pytest
from discord.ext import commands
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from src.utils import ErrorHandler
import os
from src.discord_bot.commands import BaseCommand, PingCommand, CommandRegistry
from config.settings import Config, BotConfig
from src.discord_bot.bot import AIChatBot

# Command Tests
async def test_ping_command():
    """Test that ping command sends 'Pong!' response"""
    ctx = MagicMock()
    ctx.send = AsyncMock()
    command = PingCommand()
    
    await command.execute(ctx)
    ctx.send.assert_called_once_with('Pong!')

async def test_base_command():
    """Test that BaseCommand cannot be instantiated directly"""
    with pytest.raises(TypeError) as excinfo:
        command = BaseCommand()
    
    # Verify the error message indicates it's due to being an abstract class
    assert "Can't instantiate abstract class BaseCommand with abstract method execute" in str(excinfo.value)

# Registry Tests
def test_command_registry():
    """Test command registration and retrieval"""
    registry = CommandRegistry()
    command = PingCommand()
    
    # Test registration and retrieval
    registry.register('ping', command)
    assert registry.get_command('ping') == command
    
    # Test missing command returns None
    assert registry.get_command('nonexistent') is None

def test_command_registry_duplicate():
    """Test registering duplicate commands"""
    registry = CommandRegistry()
    command1 = PingCommand()
    command2 = PingCommand()
    
    registry.register('ping', command1)
    registry.register('ping', command2)  # Should override
    assert registry.get_command('ping') == command2

# Config Tests
def test_config_loading():
    """Test default config loading"""
    with patch.dict(os.environ, {'DISCORD_TOKEN': 'test-token'}):
        config = Config.load()
        assert config.bot.command_prefix == '!'  # Default value
        assert config.bot.token == 'test-token'

def test_config_from_env():
    """Test config loading from environment variables"""
    with patch.dict(os.environ, {
        'DISCORD_TOKEN': 'test-token',
        'COMMAND_PREFIX': '$'
    }):
        config = Config.load()
        assert config.bot.command_prefix == '$'
        assert config.bot.token == 'test-token'

def test_config_defaults():
    """Test config default values"""
    with patch.dict(os.environ, {'DISCORD_TOKEN': 'test-token'}):
        config = Config.load()
        assert config.bot.command_prefix == '!'  # Should use default

def test_bot_config_creation():
    """Test BotConfig creation directly"""
    with patch.dict(os.environ, {
        'DISCORD_TOKEN': 'test-token',
        'COMMAND_PREFIX': '?'
    }):
        bot_config = BotConfig.from_env()
        assert bot_config.token == 'test-token'
        assert bot_config.command_prefix == '?'

# Bot Tests
async def test_bot_initialization():
    """Test bot initialization with config"""
    config = MagicMock()
    config.bot.command_prefix = '!'
    
    # Create default intents
    intents = discord.Intents.default()
    bot = AIChatBot(config, intents=intents)
    
    assert bot.command_prefix == '!'
    assert isinstance(bot.command_registry, CommandRegistry)
    assert bot._config == config

@pytest.mark.asyncio
async def test_bot_ping_command():
    """Test bot ping command integration"""
    config = MagicMock()
    config.bot.command_prefix = '!'
    intents = discord.Intents.default()
    bot = AIChatBot(config, intents=intents)
    
    ctx = MagicMock()
    ctx.send = AsyncMock()
    
    # Get the command from the registry
    ping_command = bot.command_registry.get_command('ping')
    assert ping_command is not None
    await ping_command.execute(ctx)
    
    ctx.send.assert_called_once_with('Pong!')

@pytest.mark.asyncio
async def test_bot_command_integration():
    """Test full command integration through bot"""
    config = MagicMock()
    config.bot.command_prefix = '!'
    intents = discord.Intents.default()
    bot = AIChatBot(config, intents=intents)
    
    ctx = MagicMock()
    ctx.send = AsyncMock()
    
    class TestCommand(BaseCommand):
        async def execute(self, ctx):
            await ctx.send("Test executed!")
    
    # Register and test through the registry
    test_command = TestCommand()
    bot.command_registry.register('test', test_command)
    retrieved_command = bot.command_registry.get_command('test')
    assert retrieved_command is not None
    await retrieved_command.execute(ctx)
    
    ctx.send.assert_called_once_with("Test executed!")