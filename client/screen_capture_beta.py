import numpy as np
import cv2
import asyncio
import websockets
import base64
import threading
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui
import json
import pyttsx3
import screeninfo  # For multi-monitor support
import av
import time
import platform
import queue

class AudioVideoCapture:
    def __init__(self, width=None, height=None, fps=30, audio_rate=44100, monitor_index=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_rate = audio_rate
        self.monitor_index = monitor_index
        
        # Determine platform-specific settings
        self.system = platform.system()
        self.configure_capture_settings()
        
        # Queues for frames
        self.video_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=20)
        
        # Capture thread
        self.capture_thread = None
        self.running = False
        
    def configure_capture_settings(self):
        """Configure capture settings based on platform"""
        # Get screen dimensions if not specified
        if self.width is None or self.height is None:
            monitors = screeninfo.get_monitors()
            if self.monitor_index >= len(monitors):
                raise ValueError(f"Monitor index {self.monitor_index} out of range")
                
            monitor = monitors[self.monitor_index]
            self.width = monitor.width
            self.height = monitor.height
        
        # Platform-specific settings
        if self.system == "Windows":
            self.video_input = f"gdigrab"
            self.video_options = {
                'framerate': str(self.fps),
                'video_size': f'{self.width}x{self.height}',
                'offset_x': '0',
                'offset_y': '0',
                'draw_mouse': '1'
            }
            self.audio_input = "dshow"
            self.audio_options = {
                'audio_buffer_size': '50'
            }
            self.audio_device = "audio=virtual-audio-capturer"
            
        elif self.system == "Linux":
            self.video_input = "x11grab"
            self.video_options = {
                'framerate': str(self.fps),
                'video_size': f'{self.width}x{self.height}',
                'grab_x': '0',
                'grab_y': '0'
            }
            self.audio_input = "pulse"
            self.audio_options = {}
            self.audio_device = "default"
            
        elif self.system == "Darwin":  # macOS
            self.video_input = "avfoundation"
            self.video_options = {
                'framerate': str(self.fps),
                'video_size': f'{self.width}x{self.height}'
            }
            self.audio_input = "avfoundation"
            self.audio_options = {}
            self.audio_device = "1:0"  # Default audio device
        else:
            raise ValueError(f"Unsupported platform: {self.system}")
            
    def start_capture(self):
        """Start the capture process"""
        if self.running:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def _capture_thread(self):
        """Thread function that captures audio and video frames"""
        try:
            # Start video capture
            video_container = av.open(
                f"{self.video_input}:{self.monitor_index}",
                options=self.video_options,
                format=self.video_input
            )
            
            # Start audio capture
            audio_container = av.open(
                f"{self.audio_input}:{self.audio_device}", 
                options=self.audio_options,
                format=self.audio_input
            )
            
            # Get streams
            video_stream = video_container.streams.video[0]
            video_stream.thread_type = 'AUTO'
            
            audio_stream = audio_container.streams.audio[0]
            audio_stream.thread_type = 'AUTO'
            
            # Set timebase for synchronization
            base_time = time.time()
            
            # Main capture loop
            while self.running:
                # Capture video frame
                for packet in video_container.demux(video_stream):
                    if not self.running:
                        break
                        
                    for frame in packet.decode():
                        try:
                            # Convert to numpy array (BGR format for OpenCV compatibility)
                            img = frame.to_ndarray(format='bgr24')
                            timestamp = time.time() - base_time
                            
                            if not self.video_queue.full():
                                self.video_queue.put((timestamp, img), block=False)
                            break
                        except queue.Full:
                            # Queue is full, skip frame
                            pass
                    break  # Process only one packet at a time
                
                # Capture audio frame
                for packet in audio_container.demux(audio_stream):
                    if not self.running:
                        break
                        
                    for frame in packet.decode():
                        try:
                            # Get audio data
                            audio_data = frame.to_ndarray()
                            timestamp = time.time() - base_time
                            
                            if not self.audio_queue.full():
                                self.audio_queue.put((timestamp, audio_data), block=False)
                            break
                        except queue.Full:
                            # Queue is full, skip frame
                            pass
                    break  # Process only one packet at a time
                
                # Brief sleep to prevent CPU overuse
                time.sleep(1.0 / (self.fps * 2))
                
        except Exception as e:
            print(f"Error in capture thread: {e}")
        finally:
            self.running = False
            try:
                video_container.close()
                audio_container.close()
            except:
                pass
    
    def capture(self):
        """Get the latest video frame (compatible with existing ScreenCapture interface)"""
        if not self.running:
            return None
            
        try:
            # Non-blocking to get the latest frame
            timestamp, frame = self.video_queue.get(block=False)
            return frame
        except queue.Empty:
            return None
    
    def get_audio(self):
        """Get the latest audio sample"""
        if not self.running:
            return None, 0
            
        try:
            # Non-blocking to get the latest audio
            timestamp, audio_data = self.audio_queue.get(block=False)
            return audio_data, timestamp
        except queue.Empty:
            return None, 0
    
    def get_synced_frame(self):
        """Get synchronized video and audio"""
        if not self.running:
            return None, None, 0
            
        # First get the latest video frame
        try:
            video_timestamp, video_frame = self.video_queue.get(block=False)
        except queue.Empty:
            return None, None, 0
            
        # Find the closest audio frame
        closest_audio = None
        closest_timestamp = 0
        closest_diff = float('inf')
        
        # Check all available audio frames
        audio_frames = []
        try:
            while not self.audio_queue.empty():
                audio_timestamp, audio_data = self.audio_queue.get(block=False)
                audio_frames.append((audio_timestamp, audio_data))
                
                # Calculate time difference
                diff = abs(audio_timestamp - video_timestamp)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_audio = audio_data
                    closest_timestamp = audio_timestamp
        except queue.Empty:
            pass
            
        # Put back audio frames we didn't use (except the one we're using)
        for ts, data in audio_frames:
            if ts != closest_timestamp:
                try:
                    self.audio_queue.put((ts, data), block=False)
                except queue.Full:
                    pass
                    
        return video_frame, closest_audio, video_timestamp
        
    def stop_capture(self):
        """Stop the capture process"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)

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
        # Get monitor information using screeninfo
        monitors = screeninfo.get_monitors()
        
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} out of range")
            
        self.monitor = monitors[monitor_index]
        
        # Set dimensions based on selected monitor
        self.left = self.monitor.x
        self.top = self.monitor.y
        self.width = self.monitor.width
        self.height = self.monitor.height
        print(f"Screen dimensions: {self.width}x{self.height}")
        
    def capture(self):
        try:
            # Capture screen region using pyautogui
            screenshot = pyautogui.screenshot(region=(
                self.left, self.top, self.width, self.height
            ))
            
            # Convert PIL Image to numpy array
            img = np.array(screenshot)
            
            # Convert from RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

def print_message(message):
    print(f"Message: {message}", end="\r")

class StreamingClient:
    def __init__(self, root, server_url="ws://localhost:8000/ws", monitor_index=0):
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
        self.processor_id = "screen_capture"  # Added processor_id property
        
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

        # GUI sliders for proper cropping
        self.left_pane = tk.Label(self.control_frame, text="Left Pane:")
        self.left_pane.pack(side='left', padx=5)
        self.left_pane = tk.Scale(self.control_frame, from_=0, to=400, 
                                     orient='vertical', command=self.update_left)
        self.left_pane.set(self.crop[0])
        self.left_pane.pack(side='left', padx=5)

        self.right_pane = tk.Label(self.control_frame, text="Right Pane:")
        self.right_pane.pack(side='left', padx=5)
        self.right_pane = tk.Scale(self.control_frame, from_=0, to=400, 
                                     orient='vertical', command=self.update_right)
        self.right_pane.set(self.crop[1])
        self.right_pane.pack(side='left', padx=5)

        self.top_pane = tk.Label(self.control_frame, text="Top Pane:")
        self.top_pane.pack(side='left', padx=5)
        self.top_pane = tk.Scale(self.control_frame, from_=0, to=300, 
                                     orient='vertical', command=self.update_top)
        self.top_pane.set(self.crop[2])
        self.top_pane.pack(side='left', padx=5)

        self.bottom_pane = tk.Label(self.control_frame, text="Bottom Pane:")
        self.bottom_pane.pack(side='left', padx=5)
        self.bottom_pane = tk.Scale(self.control_frame, from_=0, to=300, 
                                     orient='vertical', command=self.update_bottom)
        self.bottom_pane.set(self.crop[3])
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
            
            # Clean up
            self.stream_thread = None
            self.current_task = None
            self.websocket = None
            self.loop = None