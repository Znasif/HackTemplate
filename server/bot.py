# This example requires the 'message_content' intent.

import discord
from discord import Button, ButtonStyle
from discord.ui import View
from discord.ext import commands
import os
import base64
from dotenv import load_dotenv
import subprocess
import httpx
import asyncio, io
from utils import render_code, encode_image

class LlamaCppServerModifier:
    def __init__(self, model_path, port=8080, host='127.0.0.1'):
        """
        Initialize the Llama.cpp server modifier with async support.
        
        :param model_path: Path to the GGUF model file
        :param port: Port to run the server on
        :param host: Host address for the server
        """
        self.model_path = model_path
        self.port = port
        self.host = host
        self.server_process = None
        self.client = None
    
    async def start_server(self):
        """
        Asynchronously start the llama.cpp server.
        """
        # Construct the server launch command
        server_command = [
            '/home/znasif/llama.cpp/build/bin/llama-server',  # Assumes llama-server is in PATH
            '-m', self.model_path,
            '--host', str(self.host),
            '--port', str(self.port)
        ]
        
        # Launch the server as a subprocess
        # self.server_process = subprocess.Popen(
        #     server_command, 
        #     stdout=subprocess.PIPE, 
        #     stderr=subprocess.PIPE,
        #     text=True
        # )
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Wait for server to be ready with exponential backoff
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                response = await self._test_server_connection()
                if response:
                    print(f"Llama.cpp server started successfully on {self.host}:{self.port}")
                    return
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
        
        raise RuntimeError("Could not connect to the server after multiple attempts")
    
    async def _test_server_connection(self):
        """
        Async test connection to the server.
        
        :return: True if server is responsive, False otherwise
        """
        try:
            response = await self.client.get(f'http://{self.host}:{self.port}/health')
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False
    
    async def modify_text(self, 
                    original_text, 
                    instruction="Rewrite the text to be more concise",
                    max_tokens=150,
                    temperature=0.7):
        """
        Modify text using the running llama.cpp server asynchronously.
        
        :param original_text: Text to be modified
        :param instruction: Specific instruction for text modification
        :param max_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature for text generation
        :return: Modified text
        """
        # Construct the full prompt
        full_prompt = full_prompt = f"""<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{original_text}<|im_end|>\n<|im_start|>assistant"""
        #f"{instruction}\nQuery Text: {original_text}\nResponse Text:"
        print(f"Received prompt: {full_prompt}")
        # Prepare request payload
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["<|im_end|>"]
        }
        
        #try:
            # Send async request to the server
        response = await self.client.post(
            f'http://{self.host}:{self.port}/completion', 
            json=payload
        )
        #response = await self.client.get(f'http://{self.host}:{self.port}/health')
        # Check if request was successful
        if response.status_code == 200:
            # Extract the generated text
            result = response.json()
            #modified_text = result.get('content', '').strip()
            print(f"Modified text: {result}")
            return result["content"]
        else:
            print(f"Server error: {response.status_code}")
            return None
        
        # except (httpx.RequestError, httpx.HTTPStatusError) as e:
        #     print(f"Request error details: {str(e)}")
        #     print(f"Error type: {type(e)}")
        #     print(f"URL attempted: http://{self.host}:{self.port}/completion")
        #     print(f"Payload attempted: {payload}")
        #     return None
    
    async def _stop_server(self):
        """
        Async method to stop the llama.cpp server.
        """
        if self.client:
            await self.client.aclose()
        
        if self.server_process:
            print("Stopping llama.cpp server...")
            self.server_process.terminate()
            try:
                # Wait for the process to end
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                self.server_process.kill()
            
            print("Server stopped.")
    
    async def __aenter__(self):
        """
        Async context manager entry - start the server.
        """
        #await self.start_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit - stop the server.
        """
        #await self._stop_server()

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
                instruction="Provide clear, concise, and working matplotlib code examples with absolutely no greetings and commentary. Your response should be fully executable and without needing additional parsing.",
                max_tokens=500,
                temperature=0.7
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
    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        guild = discord.Object(id=int(os.getenv('DISCORD_GUILD_ID')))
        #try:
        @self.tree.command(
            name="query",
            description="Query the Llama.cpp server",
            guild=guild
        )
        async def queryLlama(interaction: discord.Interaction, prompt: str):
            #try:
                await interaction.response.defer()
                
                # Get the code response from the LLM
                code_response = await self.modifier.modify_text(
                    prompt, 
                    instruction="Provide clear, concise, and working matplotlib code examples with absolutely no greetings and commentary. Your response should be fully executable and without needing additional parsing.",
                    max_tokens=150,
                    temperature=0.7
                )
                
                # Render the code to get the image
                image = render_code(code_response)
                
                if image:
                    # Use encode_image to convert to base64
                    encoded_image = encode_image(image)
                    
                    # Convert base64 string back to bytes for Discord
                    image_bytes = base64.b64decode(encoded_image)
                    image_binary = io.BytesIO(image_bytes)
                    
                    view = ContinueView(code_response, modifier=self.modifier)
                    
                    # Send both code and image
                    await interaction.followup.send(
                        file=discord.File(fp=image_binary, filename='plot.png')
                    )

                    # Send TTS message with replay button
                    await interaction.followup.send(
                        content=code_response, 
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
                    
            # except Exception as e:
            #     if not interaction.response.is_done():
            #         await interaction.response.send_message(
            #             "Sorry, there was an error processing your request.",
            #             ephemeral=True
            #         )
            #     else:
            #         await interaction.followup.send(
            #             "Sorry, there was an error processing your request.",
            #             ephemeral=True
            #         )
            #     print(f"Error in query command: {e}")

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
        token=os.getenv('DISCORD_TOKEN')
        await client.start(token)

if __name__ == "__main__":
    asyncio.run(main())
