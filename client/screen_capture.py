import numpy as np
import cv2
import asyncio
import websockets
import base64
import threading
import tkinter as tk
from PIL import Image, ImageTk
import win32gui
import win32ui
import win32con
import win32api
import json
import threading
import pyttsx3
import queue
import pyaudio
import wave
import struct
import time
import audioop

class TTSHandler:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.current_text = None
        self.stop_flag = threading.Event()
        self.tts_thread = None

    def speak_text(self, text):
        """Stop current speech and start new text"""
        # Stop current speech if any
        if self.tts_thread and self.tts_thread.is_alive():
            self.stop_flag.set()
            self.tts_thread.join(timeout=0.1)  # Brief wait for cleanup
            
        # Reset stop flag and start new speech
        self.stop_flag.clear()
        self.current_text = text
        self.tts_thread = threading.Thread(target=self._speak_thread)
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def _speak_thread(self):
        try:
            # Create a new engine instance for this thread
            thread_engine = pyttsx3.init()
            thread_engine.say(self.current_text)
            thread_engine.startLoop(False)
            while not self.stop_flag.is_set():
                thread_engine.iterate()
            thread_engine.endLoop()
        except Exception as e:
            print(f"TTS error: {e}")

class ScreenCapture:
    def __init__(self, monitor_index):
        # get screen size
        # self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        # self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        # Get monitor info
        monitors = win32api.EnumDisplayMonitors()
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} out of range")
            
        monitor = monitors[monitor_index]
        monitor_rect = monitor[2]  # monitor[2] contains (left, top, right, bottom)
        
        # Set dimensions based on selected monitor
        self.left = monitor_rect[0]
        self.top = monitor_rect[1]
        self.width = monitor_rect[2] - monitor_rect[0]
        self.height = monitor_rect[3] - monitor_rect[1]
        print(f"Screen dimensions: {self.width}x{self.height}")
        
    def capture(self):
        try:
            # create a DC for the entire virtual screen
            hdesktop = win32gui.GetDesktopWindow()
            desktop_dc = win32gui.GetWindowDC(hdesktop)
            img_dc = win32ui.CreateDCFromHandle(desktop_dc)
            
            # create a memory DC
            mem_dc = img_dc.CreateCompatibleDC()
            
            # create a bitmap object
            screenshot = win32ui.CreateBitmap()
            screenshot.CreateCompatibleBitmap(img_dc, self.width, self.height)
            mem_dc.SelectObject(screenshot)
            
            # copy screen into memory DC
            mem_dc.BitBlt((0, 0), (self.width, self.height), img_dc, 
                        (self.left, self.top), win32con.SRCCOPY)
            
            # convert bitmap to numpy array
            signedIntsArray = screenshot.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (self.height, self.width, 4)
            
            # convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Cleanup
            mem_dc.DeleteDC()
            win32gui.DeleteObject(screenshot.GetHandle())
            img_dc.DeleteDC()
            win32gui.ReleaseDC(hdesktop, desktop_dc)
            
            return img
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

def print_message(message):
    print(f"Message: {message}", end="\r")

class AudioStreamer:
    def __init__(self, server_url="ws://localhost:8000/audio", audio_device_index=None):
        self.server_url = server_url
        self.running = False
        self.websocket = None
        self.loop = None
        self.current_task = None
        self.stream_thread = None
        self.device_index = audio_device_index  # The audio device index to use
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 16kHz required by Whisper
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = queue.Queue()
        self.transcription_callback = None
        
        # Print available audio devices
        self.print_audio_devices()
        
    def print_audio_devices(self):
        """Print all available audio input devices"""
        print("\nAvailable audio input devices:")
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        input_devices = []
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # Input device
                input_devices.append((i, device_info.get('name')))
                print(f"  [{i}] {device_info.get('name')}")
        
        if self.device_index is None and input_devices:
            # Use first available input device if none specified
            self.device_index = input_devices[0][0]
            print(f"\nUsing default input device: [{self.device_index}] {input_devices[0][1]}")
        elif self.device_index is not None:
            device_name = "Unknown"
            try:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                device_name = device_info.get('name')
            except:
                pass
            print(f"\nUsing specified input device: [{self.device_index}] {device_name}")
        
    def set_transcription_callback(self, callback):
        """Set a callback function to handle received transcriptions"""
        self.transcription_callback = callback
    
    def start_audio_stream(self):
        """Start audio streaming in a separate thread"""
        if not self.running and (self.stream_thread is None or not self.stream_thread.is_alive()):
            self.running = True
            self.stream_thread = threading.Thread(target=self._run_audio_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            print("Audio streaming started")
    
    def stop_audio_stream(self):
        """Stop audio streaming"""
        if self.running:
            self.running = False
            
            # Cancel the current task
            if self.current_task and not self.current_task.done():
                if self.loop and not self.loop.is_closed():
                    self.loop.call_soon_threadsafe(self.current_task.cancel)
            
            # Wait for thread to finish with timeout
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5)
                if self.stream_thread.is_alive():
                    print("Warning: Audio stream thread did not terminate properly")
            
            # Close audio stream if open
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            # Clean up
            self.stream_thread = None
            self.current_task = None
            self.websocket = None
            print("Audio streaming stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.running:
            self.audio_buffer.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _run_audio_loop(self):
        """Run the audio streaming loop in a separate thread"""
        try:
            # Open audio stream
            try:
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk,
                    stream_callback=self._audio_callback
                )
                
                # Start the stream
                self.stream.start_stream()
                print("Audio stream started successfully")
            except Exception as e:
                print(f"Error opening audio stream: {str(e)}")
                print("Make sure a microphone is connected and accessible")
                self.running = False
                return
            
            # Create and run async loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Create and store the task
            self.current_task = self.loop.create_task(self._process_audio())
            
            # Run until task is complete or cancelled
            self.loop.run_until_complete(self.current_task)
        except Exception as e:
            print(f"Error in audio loop: {e}")
        finally:
            # Close the audio stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            # Close the loop
            if self.loop and not self.loop.is_closed():
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self.loop.close()
                except Exception as e:
                    print(f"Error closing audio loop: {e}")
            
            self.loop = None
            self.current_task = None
    
    async def _process_audio(self):
        """Process audio data and send to server"""
        try:
            # Connect to WebSocket
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                print(f"Connected to audio server at {self.server_url}")
                
                # Process audio chunks
                audio_data = bytearray()
                last_send_time = time.time()
                send_interval = 3.0  # Send every 3 seconds
                
                while self.running:
                    # Get audio chunk from buffer (with timeout)
                    try:
                        chunk = self.audio_buffer.get(timeout=0.1)
                        audio_data.extend(chunk)
                    except queue.Empty:
                        # No new audio data, check if it's time to send what we have
                        pass
                    
                    current_time = time.time()
                    # Send data if we have enough or enough time has passed
                    if len(audio_data) > 0 and (current_time - last_send_time) >= send_interval:
                        try:
                            # Convert audio to float32 for Whisper
                            pcm_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # Prepare the audio data - send as a simple list of floats (no base64 encoding)
                            message = {
                                "audio": pcm_data.tolist()  # Convert NumPy array to list for JSON
                            }
                            
                            # Send to server
                            await websocket.send(json.dumps(message))
                            print(f"Sent {len(pcm_data)} audio samples")
                            
                            # Get transcription result
                            response = await websocket.recv()
                            response_data = json.loads(response)
                            
                            if "text" in response_data and response_data["text"]:
                                transcription = response_data["text"]
                                print(f"Received transcription: {transcription}")
                                
                                # Call the callback function if set
                                if self.transcription_callback:
                                    self.transcription_callback(transcription)
                        except Exception as e:
                            print(f"Error sending/receiving audio data: {e}")
                        finally:
                            # Always reset for next interval
                            audio_data = bytearray()
                            last_send_time = current_time
                    
                    # Sleep briefly to prevent high CPU usage
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print("Audio processing task cancelled")
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            self.websocket = None
    
    def __del__(self):
        """Clean up resources"""
        self.stop_audio_stream()
        if self.audio:
            self.audio.terminate()

class StreamingClient:
    def __init__(self, root, server_url="ws://localhost:8000/ws", monitor_index=0, audio_device_index=None):
        self.root = root
        self.server_url = server_url
        self.running = False
        self.screen_capture = ScreenCapture(monitor_index)
        self.websocket = None
        self.loop = None
        self.current_task = None
        self.stream_thread = None  # Add explicit thread tracking
        self.max_dimension = (853, 480)  # Maximum dimensions for frames
        self.jpeg_quality = 30  # JPEG compression quality (1-100)
        self.crop = [0, 0, 0, 0]
        self.setup_gui()
        self.tts_handler = TTSHandler()
        self.received_text = ""
        
        # Create audio streamer for speech recognition
        self.audio_streamer = AudioStreamer(audio_device_index=audio_device_index)
        self.audio_streamer.set_transcription_callback(self.handle_transcription)
        
    def handle_transcription(self, text):
        """Handle received transcription from audio stream"""
        if text != self.received_text:
            self.received_text = text
            self.tts_handler.speak_text(text)  # Optionally speak the transcription
            # Update the UI if needed
        
    def setup_gui(self):
        # Create frame for image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(expand=True, fill='both')
        self.image_frame.pack_propagate(False)
        # Create label for image
        self.label = tk.Label(self.image_frame)
        self.label.pack(expand=True, fill='both')
        
        # Create frame for controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill='x')
        
        # Add FPS counter
        self.fps_label = tk.Label(self.control_frame, text="FPS: 0")
        self.fps_label.pack(side='left', padx=5)
        
        # Add quality slider
        self.quality_label = tk.Label(self.control_frame, text="Quality:")
        self.quality_label.pack(side='left', padx=5)
        self.quality_slider = tk.Scale(self.control_frame, from_=1, to=100, 
                                     orient='horizontal', command=self.update_quality)
        self.quality_slider.set(self.jpeg_quality)
        self.quality_slider.pack(side='left', padx=5)


        '''gui sliders for propper cropping'''
        self.left_pane = tk.Label(self.control_frame, text="Left Pane:")
        self.left_pane.pack(side='left', padx=5)
        self.left_pane = tk.Scale(self.control_frame, from_=0, to=400, 
                                     orient='vertical', command=self.update_left)
        self.left_pane.set(self.jpeg_quality)
        self.left_pane.pack(side='left', padx=5)

        self.right_pane = tk.Label(self.control_frame, text="Right Pane:")
        self.right_pane.pack(side='left', padx=5)
        self.right_pane = tk.Scale(self.control_frame, from_=0, to=400, 
                                     orient='vertical', command=self.update_right)
        self.right_pane.set(self.jpeg_quality)
        self.right_pane.pack(side='left', padx=5)

        self.top_pane = tk.Label(self.control_frame, text="Top Pane:")
        self.top_pane.pack(side='left', padx=5)
        self.top_pane = tk.Scale(self.control_frame, from_=0, to=300, 
                                     orient='vertical', command=self.update_top)
        self.top_pane.set(self.jpeg_quality)
        self.top_pane.pack(side='left', padx=5)

        self.bottom_pane = tk.Label(self.control_frame, text="Bottom Pane:")
        self.bottom_pane.pack(side='left', padx=5)
        self.bottom_pane = tk.Scale(self.control_frame, from_=0, to=300, 
                                     orient='vertical', command=self.update_bottom)
        self.bottom_pane.set(self.jpeg_quality)
        self.bottom_pane.pack(side='left', padx=5)

        self.last_frame_time = 0
        
    def update_quality(self, value):
        self.jpeg_quality = int(value)
    
    def update_left(self, left_value):
        self.crop[0]=int(left_value)
    
    def update_right(self, right_value):
        self.crop[1]=int(right_value)
    
    def update_top(self, top_value):
        self.crop[2]=int(top_value)
    
    def update_bottom(self, bottom_value):
        self.crop[3]=int(bottom_value)
        
    def compress_frame(self, frame):
        # Resize if larger than max_dimension
        height, width = frame.shape[:2]
        if width > self.max_dimension[0] or height > self.max_dimension[1]:
            ratio = min(self.max_dimension[0]/width, self.max_dimension[1]/height)
            new_size = (int(width * ratio), int(height * ratio))
            frame = cv2.resize(frame, new_size)
        
        frame = self.crop_frame(frame)
        # Compress using JPEG encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer
    
    def crop_frame(self, frame):
        """
        Crops a given frame (NumPy array) based on the provided left, right, top, and bottom values.
        Includes checks to ensure a valid crop is performed.

        Args:
            frame (np.ndarray): The input frame (e.g., from screen capture).
            left (int): Number of pixels to remove from the left edge.
            right (int): Number of pixels to remove from the right edge.
            top (int): Number of pixels to remove from the top edge.
            bottom (int): Number of pixels to remove from the bottom edge.

        Returns:
            np.ndarray or None: The cropped frame if the cropping parameters are valid,
                                otherwise None.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            print("Error: Input frame is None or not a NumPy array.")
            return None

        height, width = frame.shape[:2]  # Get height and width
        left, right, top, bottom = self.crop
        # Check for invalid cropping values
        if left < 0 or right < 0 or top < 0 or bottom < 0:
            print("Error: Cropping values cannot be negative.")
            return None

        # Calculate new dimensions
        new_left = left
        new_top = top
        new_right = width - right
        new_bottom = height - bottom

        # Check if the new dimensions are valid
        if new_left >= new_right or new_top >= new_bottom:
            print("Error: Invalid cropping parameters resulted in a non-positive width or height.")
            return None

        # Perform the cropping
        try:
            cropped_frame = frame[new_top:new_bottom, new_left:new_right]
            return cropped_frame
        except IndexError:
            print("Error: Index out of bounds during cropping. Check your cropping values.")
            return None
    
    def update_image(self, image):
        try:
            # Resize image to fit window while maintaining aspect ratio
            window_width = self.image_frame.winfo_width()
            window_height = self.image_frame.winfo_height()
            
            if window_width > 0 and window_height > 0:
                img_aspect = image.shape[1] / image.shape[0]
                window_aspect = window_width / window_height
                
                if window_aspect > img_aspect:
                    new_height = window_height
                    new_width = int(window_height * img_aspect)
                else:
                    new_width = window_width
                    new_height = int(window_width / img_aspect)
                
                image = cv2.resize(image, (new_width, new_height))
            
            #print(image.shape)
            # Convert to PhotoImage
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            self.label.config(image=photo)
            self.label.image = photo
            
            # Update FPS
            current_time = asyncio.get_event_loop().time()
            if self.last_frame_time > 0:
                fps = 1 / (current_time - self.last_frame_time)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"Error updating image: {e}")

    async def capture_and_stream(self):
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                while self.running:
                    try:
                        # Capture screen
                        frame = self.screen_capture.capture()
                        if frame is None:
                            continue
                        # Compress frame
                        compressed = self.compress_frame(frame)
                        #print(compressed.shape)
                        img_str = base64.b64encode(compressed).decode('utf-8')
                        
                        # Send frame with processor ID
                        message = {
                            "image": f"data:image/jpeg;base64,{img_str}",
                            "processor": self.processor_id  # Use the instance property
                        }
                        await websocket.send(json.dumps(message))
                        
                        # Receive processed frame
                        response = await websocket.recv()
                        data = json.loads(response)
                        # Decode and display processed frame
                        if "image" in data:
                            encoded_data = data["image"].split(',')[1]
                            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            self.root.after(0, self.update_image, processed_frame)
                        if "text" in data:
                            print_message(f"Received text: {data['text']}")
                            print_message(f"Previous text: {self.received_text}")
                            print_message(f"Are they different? {data['text'] != self.received_text}")
                            if data["text"] != self.received_text:
                                self.received_text = data["text"]
                                self.root.after(0, self.tts_handler.speak_text, data["text"])
                        
                        await asyncio.sleep(1/30)  # Limit to 30 FPS
                        
                    except asyncio.CancelledError:
                        raise  # Re-raise to handle cancellation
                    except Exception as e:
                        print(f"Error during streaming: {e}")
                        await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Streaming task cancelled")
        finally:
            self.websocket = None
    
    def start(self):
        if not self.running and (self.stream_thread is None or not self.stream_thread.is_alive()):
            self.running = True
            self.stream_thread = threading.Thread(target=self._run_async_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
        else:
            print("Stream already running or thread still alive")
    
    def _run_async_loop(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Create and store the task
            self.current_task = self.loop.create_task(self.capture_and_stream())
            
            # Run until task is complete or cancelled
            self.loop.run_until_complete(self.current_task)
        except asyncio.CancelledError:
            print("Main task cancelled")
        except Exception as e:
            print(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                    try:
                        self.loop.run_until_complete(self.current_task)
                    except (asyncio.CancelledError, Exception) as e:
                        print(f"Task cancellation: {e}")
                
                # Close the loop
                if self.loop and not self.loop.is_closed():
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self.loop.close()
            except Exception as e:
                print(f"Cleanup error: {e}")
            finally:
                self.loop = None
                self.current_task = None
    
    def stop(self):
        if self.running:
            self.running = False
            
            # Cancel the current task
            if self.current_task and not self.current_task.done():
                if self.loop and not self.loop.is_closed():
                    self.loop.call_soon_threadsafe(self.current_task.cancel)
            
            # Wait for thread to finish with timeout
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5)
                if self.stream_thread.is_alive():
                    print("Warning: Stream thread did not terminate properly")
            
            # Stop audio streaming if it's running
            if hasattr(self, 'audio_streamer'):
                self.audio_streamer.stop_audio_stream()
            
            # Clean up
            self.stream_thread = None
            self.current_task = None
            self.websocket = None
            self.loop = None