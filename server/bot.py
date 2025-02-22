# This example requires the 'message_content' intent.

import discord
from discord import Button, ButtonStyle
from discord.ui import View
from discord.ext import commands
import os
import base64
from dotenv import load_dotenv
from llama import LlamaCppServerModifier
from llava import ImageDescriber
import asyncio, io
from utils import render_code, encode_image
import httpx

class ContinueView(View):
    active_threads = {}  # Class variable to store active thread views

    def __init__(self, first_prompt: str, modifier):
        super().__init__(timeout=None)
        self.prompts = [first_prompt]
        self.thread = None
        self.modifier = modifier

    @discord.ui.button(label="Continue", style=ButtonStyle.secondary)
    async def continue_query(self, interaction: discord.Interaction, button: Button):
        if not self.thread:
            self.thread = await interaction.message.create_thread(
                name="Continue",
                auto_archive_duration=60
            )
            print(f"Created thread: {self.thread.id}")
            # Store this view instance in the class dictionary
            ContinueView.active_threads[self.thread.id] = self
            await interaction.response.send_message("Created a thread to continue query", ephemeral=True)
        
        if not interaction.response.is_done():
            await interaction.response.defer()
    
    async def handle_thread_message(self, message):
        # Add new message to prompts context
        self.prompts.append(message.content)
        
        # Create combined prompt with context
        combined_prompt = " ".join(self.prompts)
        print(f"Combined prompt: {combined_prompt}")
        try:
            # Get the code response from the LLM
            code_response = await self.modifier.modify_text(
                combined_prompt, 
                instruction="""Provide working python code that:
1. If plotting, use matplotlib directly
2. If drawing complex shapes or geometry, use PIL or other image libraries, but convert the image to a matplotlib figure before displaying
3. Always end with plt.show()
4. Include all necessary imports
5. Use no text output, comments, or explanations
6. Be fully self-contained and executable
7. Never use Image.show() or other display methods
8. Always convert non-matplotlib images to plt format using:
   plt.imshow(image)
   plt.axis('off')
   plt.show()""",
                max_tokens=-1,
                temperature=0.8
            )

            print(f"Code response: {code_response}")
            
            # Render the code to get the image
            image = render_code(code_response)
            
            if image:
                encoded_image = encode_image(image)
                image_bytes = base64.b64decode(encoded_image)
                image_binary = io.BytesIO(image_bytes)
                
                await self.thread.send(
                    content=code_response,
                    tts=True
                )
                
                await self.thread.send(
                    file=discord.File(fp=image_binary, filename='plot.png')
                )
            else:
                await self.thread.send(
                    content=code_response + "\n*Note: Plot generation failed*",
                    tts=True
                )
                
        except Exception as e:
            await self.thread.send(
                "Sorry, there was an error processing your request."
            )
            print(f"Error in thread message processing: {e}")

class MyClient(commands.Bot):
    modifier = None
    describer = None
    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        guild = discord.Object(id=int(os.getenv('DISCORD_GUILD_ID')))

        @self.tree.command(name="queryimage",
                        description="Query with an image and prompt",
                        guild=guild)
        async def queryimage(
            interaction: discord.Interaction, 
            image: discord.Attachment,
            prompt: str
        ):
            try:
                await interaction.response.defer()
                
                if not image.content_type.startswith('image/'):
                    await interaction.followup.send("Please provide a valid image file.")
                    return
                
                # Download the image data using httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(image.url)
                    if response.status_code == 200:
                        image_data = response.content
                        # Convert to base64
                        encoded_image = base64.b64encode(image_data).decode('utf-8')
                        
                        # Now you can use encoded_image with describe_image
                        description = await self.describer.describe_with_gemma(encoded_image, prompt=prompt)
                        
                        view = ContinueView(prompt, modifier=self.modifier)
                        
                        await interaction.followup.send(
                            content=description,
                            tts=True,
                            view=view
                        )
                    else:
                        await interaction.followup.send("Failed to download the image.")
                        
            except Exception as e:
                await interaction.followup.send(f"An error occurred: {str(e)}")
        #try:
        @self.tree.command(
            name="querycode",
            description="Query the Llama.cpp server",
            guild=guild
        )
        async def queryLlama(interaction: discord.Interaction, prompt: str):
            try:
                await interaction.response.defer()
                
                # Get the code response from the LLM
                code_response = await self.modifier.modify_text(
                    prompt, 
                    instruction="""Provide working python code that:
1. If plotting, use matplotlib directly
2. If drawing complex shapes or geometry, use PIL or other image libraries, but convert the image to a matplotlib figure before displaying
3. Always end with plt.show()
4. Include all necessary imports
5. Use no text output, comments, or explanations
6. Be fully self-contained and executable
7. Never use Image.show() or other display methods
8. Always convert non-matplotlib images to plt format using:
   plt.imshow(image)
   plt.axis('off')
   plt.show()""",
                    max_tokens=-1,
                    temperature=0.8
                )
                
                # Render the code to get the image
                image = render_code(code_response, True)
                
                if image:
                    # Use encode_image to convert to base64
                    encoded_image = encode_image(image)
                    # Convert base64 string back to bytes for Discord
                    image_bytes = base64.b64decode(encoded_image)
                    image_binary = io.BytesIO(image_bytes)
                    
                    view = ContinueView(code_response, modifier=self.modifier)
                    
                    # Send both code and image
                    await interaction.followup.send(
                        content=code_response,
                        file=discord.File(fp=image_binary, filename='plot.png')
                    )

                    description = await self.describer.describe_with_qwen(encoded_image)
                    print(f"Description: {description}")
                    
                    # Send TTS message with replay button
                    await interaction.followup.send(
                        content=description,
                        tts=True,
                        view=view
                    )
                else:
                    # Send just the code if image generation failed
                    view = ContinueView(code_response, modifier=self.modifier)
                    
                    # Send TTS message with replay button
                    await interaction.followup.send(
                        content=code_response+ "\n*Note: Plot generation failed*", 
                        tts=True,
                        view=view
                    )
                    
            except Exception as e:
                if not interaction.response.is_done():
                    await interaction.response.send_message(
                        "Sorry, there was an error processing your request.",
                        ephemeral=True
                    )
                else:
                    await interaction.followup.send(
                        "Sorry, there was an error processing your request.",
                        ephemeral=True
                    )
                print(f"Error in query command: {e}")

        # Sync commands
        synced = await self.tree.sync(guild=guild)
        print(f'Synced {len(synced)} commands to guild {guild.id}')
        # except Exception as e:
        #     print(f"Error: {e}")

    async def on_message(self, message):
        if message.author == self.user:
            return
        print(f"Received message: {message.content}")
        if isinstance(message.channel, discord.Thread) and message.channel.id in ContinueView.active_threads:
            print(f"Accessing active thread: {message.channel.id}")
            view = ContinueView.active_threads[message.channel.id]
            await view.handle_thread_message(message)
    


async def main():
    load_dotenv()
    async with LlamaCppServerModifier(model_path=os.getenv('MODEL_PATH'), port=os.getenv('PORT')) as modifier:
        intents = discord.Intents.default()
        intents.message_content = True
        await modifier.start_server()
        client = MyClient(command_prefix=os.getenv('COMMAND_PREFIX'), intents=intents)
        client.modifier = modifier
        client.describer = ImageDescriber()
        token=os.getenv('DISCORD_TOKEN')
        await client.start(token)

if __name__ == "__main__":
    asyncio.run(main())
