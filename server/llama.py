import subprocess
import httpx
import asyncio

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
            "stop": ["<|im_end|>"],
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "n_probs": 0,
            "typical_p": 1.0,  # default value, adjust as needed
            "dynatemp_range": 0.0,  # dynamic temperature range
            "dynatemp_exponent": 1.0,  # dynamic temperature exponent
            "xtc_threshold": 0.1, # cross-token coherence threshold
            "xtc_probability": 0.0, # cross-token coherence probability
            "samplers":["penalties", "dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
            "repeat_last_n": 64,  # number of tokens to look back for repetition
            "repeat_penalty": 1.0,  # penalty for repetition
            "presence_penalty": 0.0,  # penalty for token presence
            "frequency_penalty": 0.0,  # penalty for token frequency
            "dry_multiplier": 0.0,  # DynamicRepetition (DRY) multiplier
            "dry_base": 1.75,  # DRY base value
            "dry_allowed_length": 2,  # DRY allowed sequence length
            "dry_penalty_last_n": -1  # DRY penalty for last N tokens
        }
        
        #try:
            # Send async request to the server
        response = await self.client.post(
            f'http://{self.host}:{self.port}/completion', 
            json=payload,
            timeout=350.0
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
