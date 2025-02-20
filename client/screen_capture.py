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

class ScreenCapture:
    def __init__(self):
        # get screen size
        # self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        # self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        # Get monitor info
        monitor_index = 1  # Primary monitor
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
                        (0, 0), win32con.SRCCOPY)
            
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

class StreamingClient:
    def __init__(self, root, server_url="ws://localhost:8000/ws"):#"ws://shaggy-cities-think.loca.lt/ws"):
        self.root = root
        self.server_url = server_url
        self.running = False
        self.screen_capture = ScreenCapture()
        self.websocket = None
        self.loop = None
        self.current_task = None
        self.stream_thread = None  # Add explicit thread tracking
        self.max_dimension = (800, 600)  # Maximum dimensions for frames
        self.jpeg_quality = 30  # JPEG compression quality (1-100)
        self.setup_gui()
        
    def setup_gui(self):
        # Create frame for image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(expand=True, fill='both')
        
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
        
        self.last_frame_time = 0
        
    def update_quality(self, value):
        self.jpeg_quality = int(value)
        
    def compress_frame(self, frame):
        # Resize if larger than max_dimension
        height, width = frame.shape[:2]
        if width > self.max_dimension[0] or height > self.max_dimension[1]:
            ratio = min(self.max_dimension[0]/width, self.max_dimension[1]/height)
            new_size = (int(width * ratio), int(height * ratio))
            frame = cv2.resize(frame, new_size)
        
        # Compress using JPEG encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer
        
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
                        
                        # Send frame
                        await websocket.send(f"data:image/jpeg;base64,{img_str}")
                        
                        # Receive processed frame
                        response = await websocket.recv()
                        
                        # Decode and display processed frame
                        encoded_data = response.split(',')[1]
                        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Update GUI in main thread
                        self.root.after(0, self.update_image, processed_frame)
                        
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