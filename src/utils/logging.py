import logging
from typing import Optional
from discord.ext import commands

def setup_logging(level: Optional[str] = None) -> None:
    logging_level = level or 'INFO'
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class ErrorHandler:
    @staticmethod
    async def handle_command_error(ctx: commands.Context, error: Exception):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("Command not found!")
        elif isinstance(error, commands.MissingPermissions):
            await ctx.send("You don't have permission to do that!")
        else:
            logging.error(f"Unexpected error: {error}", exc_info=True)
            await ctx.send("An unexpected error occurred!")