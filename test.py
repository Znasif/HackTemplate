import asyncio
import cv2
import base64
import websockets
import argparse
import numpy as np
from typing import Union
import time
import signal

def compress_frame(frame, max_dimension=(800, 600), jpeg_quality=30):
    """
    Compress frame using the same settings as the GUI client
    """
    height, width = frame.shape[:2]
    if width > max_dimension[0] or height > max_dimension[1]:
        ratio = min(max_dimension[0]/width, max_dimension[1]/height)
        new_size = (int(width * ratio), int(height * ratio))
        frame = cv2.resize(frame, new_size)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer

async def stream_to_server(source: Union[str, int], output_dir: str, fps: int = 30, duration: int = 10):
    """
    Stream video/images to WebSocket server and save processed results
    """
    # Setup video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    frame_count = 0
    start_time = time.time()
    frame_interval = 1.0 / fps
    elapsed_time = 0
    
    # Flag for graceful shutdown
    running = True
    
    def signal_handler(signum, frame):
        nonlocal running
        print("\nReceived signal to stop streaming...")
        running = False
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    websocket = None
    try:
        async with websockets.connect('ws://localhost:8000/ws') as websocket:
            print("Connected to server")
            
            while running:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Check duration limit
                if isinstance(source, str) and elapsed_time > duration:
                    print("Duration limit reached")
                    break
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        break
                
                # Control FPS
                if elapsed_time - (frame_count * frame_interval) < frame_interval:
                    await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
                    continue
                
                print(f"\nProcessing frame {frame_count}")
                print(f"Original frame shape: {frame.shape}")
                
                # Compress frame
                compressed = compress_frame(frame)
                img_base64 = base64.b64encode(compressed).decode('utf-8')
                data = f"data:image/jpeg;base64,{img_base64}"
                print(f"Compressed data length: {len(data)}")
                
                try:
                    print("Sending frame to server...")
                    await websocket.send(data)
                    print(f"Sent frame {frame_count}")
                    
                    print("Waiting for server response...")
                    response = await websocket.recv()
                    print(f"Received response length: {len(response)}")
                    
                    # Decode response
                    encoded_data = response.split(',')[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if processed_frame is not None:
                        print(f"Processed frame shape: {processed_frame.shape}")
                        output_path = f"{output_dir}/frame_{frame_count:04d}.jpg"
                        cv2.imwrite(output_path, processed_frame)
                        print(f"Saved processed frame to {output_path}")
                    else:
                        print("Failed to decode processed frame")
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    break  # Break on error to ensure clean shutdown
                
                # Rate limiting
                await asyncio.sleep(1/fps)
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if cap:
            cap.release()
        
        if websocket and not websocket.closed:
            try:
                await websocket.close()
                print("WebSocket connection closed gracefully")
            except:
                pass
        
        if 'elapsed_time' not in locals():
            elapsed_time = time.time() - start_time
        
        print(f"\nFinal Statistics:")
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        if elapsed_time > 0:
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")

async def main(args):
    try:
        # Convert input to int if it's a camera index
        try:
            source = int(args.input)
        except ValueError:
            source = args.input
        
        # Ensure output directory exists
        import os
        os.makedirs(args.output, exist_ok=True)
        
        await stream_to_server(
            source=source,
            output_dir=args.output,
            fps=args.fps,
            duration=args.duration
        )
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test WebSocket video streaming server')
    parser.add_argument('--input', '-i', required=True,
                      help='Path to input image/video file or camera index (0 for webcam)')
    parser.add_argument('--output', '-o', required=True,
                      help='Directory to save output frames')
    parser.add_argument('--fps', type=int, default=30,
                      help='Target frames per second')
    parser.add_argument('--duration', type=int, default=10,
                      help='Duration in seconds to stream (for image input)')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nExiting...")