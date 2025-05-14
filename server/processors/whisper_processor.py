from .base_processor import BaseProcessor
import numpy as np
import cv2
import subprocess
import os
import tempfile
import wave
import ctypes
from ctypes import *
import time

class WhisperProcessor(BaseProcessor):
    def __init__(self, 
                 model_path="/home/znasif/whisper.cpp/models/ggml-base.en.bin",#ggml-large-v3-turbo-q8_0.bin", 
                 language="en",
                 translate=False,
                 n_threads=4,
                 step_ms=3000,
                 length_ms=10000,
                 keep_ms=200,
                 max_tokens=32,
                 sample_rate=16000):
        """
        Initialize WhisperProcessor for speech-to-text
        
        Args:
            model_path (str): Path to the whisper model
            language (str): Language code for transcription
            translate (bool): Whether to translate to English
            n_threads (int): Number of threads to use
            step_ms (int): Audio step size in milliseconds
            length_ms (int): Audio buffer length in milliseconds
            keep_ms (int): Audio to keep from previous step
            max_tokens (int): Maximum number of tokens per audio chunk
            sample_rate (int): Audio sample rate (default 16kHz for Whisper)
        """
        super().__init__()
        
        self.model_path = model_path
        self.language = language
        self.translate = translate
        self.n_threads = n_threads
        self.step_ms = step_ms
        self.length_ms = length_ms
        self.keep_ms = keep_ms
        self.max_tokens = max_tokens
        self.sample_rate = sample_rate
        
        # Keep track of previous audio for context
        self.prev_audio = None
        self.keep_samples = int(self.keep_ms * self.sample_rate / 1000)
        
        print(f"WhisperProcessor initialized with model: {model_path}")

    def process_frame(self, frame):
        """
        Process a frame that contains audio data
        
        In this implementation, the frame is expected to contain audio data
        in its metadata or separate channel. The video frame itself is not modified.
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (unmodified_frame, transcription_text)
        """
        transcription = ""
        
        # Check if frame has audio data in metadata
        # This assumes the client is sending audio data along with the video frame
        # You'll need to adapt this part based on how your client sends audio data
        
        # For demonstration, let's assume audio data is passed as a separate property
        # in a dictionary alongside the frame. You'll need to modify this to match
        # your actual implementation.
        
        # If frame is a dictionary with "image" and "audio" keys
        if isinstance(frame, dict) and "image" in frame and "audio" in frame:
            video_frame = frame["image"]
            audio_data = frame["audio"]
            
            # Process audio data
            if audio_data is not None and len(audio_data) > 0:
                # Combine with previous audio if exists
                if self.prev_audio is not None:
                    audio_data = np.concatenate((self.prev_audio, audio_data))
                
                # Keep some audio for next time
                if len(audio_data) > self.keep_samples:
                    self.prev_audio = audio_data[-self.keep_samples:]
                else:
                    self.prev_audio = audio_data
                
                # Transcribe audio
                transcription = self.transcribe_audio(audio_data)
            
            return video_frame, transcription
        
        # If audio data is received in a different format or channel,
        # you'll need to extract it accordingly
        
        # For now, just return the frame as is with empty transcription
        return frame, transcription

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio data using whisper.cpp command-line interface
        
        Args:
            audio_data (numpy.ndarray): Audio data as float32 array
            
        Returns:
            str: Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""
            
        try:
            # Ensure audio_data is float32 and in the range [-1.0, 1.0]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save audio to a temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                # Convert float32 to int16
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            # Call whisper-cli for transcription
            whisper_cmd = [
                "/home/znasif/whisper.cpp/build/bin/whisper-cli",
                "-m", self.model_path,
                "-f", temp_file.name,
                "-l", self.language,
                "-t", str(self.n_threads)
            ]
            
            if self.translate:
                whisper_cmd.append("--translate")
            
            process = subprocess.Popen(
                whisper_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
            
            # Extract text from stdout
            transcription = ""
            for line in stdout.splitlines():
                if line and not line.startswith("["):  # Skip timestamp lines
                    transcription += line.strip() + " "
            print("New"+transcription)
            return transcription.strip()
            
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""

    def process_audio_chunk(self, audio_chunk):
        """
        Process an audio chunk directly (without video frame)
        
        This method can be used when audio is being sent separately
        from video frames, e.g. in a dedicated audio websocket connection.
        
        Args:
            audio_chunk (numpy.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        # Combine with previous audio if exists
        if self.prev_audio is not None:
            audio_data = np.concatenate((self.prev_audio, audio_chunk))
        else:
            audio_data = audio_chunk
        
        # Keep some audio for next time
        if len(audio_data) > self.keep_samples:
            self.prev_audio = audio_data[-self.keep_samples:]
        else:
            self.prev_audio = audio_data
        
        # Transcribe audio
        return self.transcribe_audio(audio_data)
