import tkinter as tk
from tkinter import ttk
from screen_capture import StreamingClient
import json

# At the top of your code, add these tracking variables
last_x = 0
last_y = 0
last_width = 0
last_height = 0

def main():
    root = tk.Tk()
    root.title("Screen Capture Client")
    root.geometry("853x480")
    
    # Set initial aspect ratio (4:3)
    aspect_ratio = 2560/1440
    min_width = 853
    min_height = int(min_width / aspect_ratio)
    
    # Make window resizable
    root.minsize(min_width, min_height)

    def enforce_aspect_ratio(event):
        global last_x, last_y, last_width, last_height
        
        # Skip if minimized
        if event.width < 1 or event.height < 1:
            return
        
        # Check if this is just a move (position changed but size didn't)
        current_x = root.winfo_x()
        current_y = root.winfo_y()
        
        is_just_move = (last_width == event.width and last_height == event.height and
                        (last_x != current_x or last_y != current_y))
        
        # Update last position
        last_x = current_x
        last_y = current_y
        last_width = event.width
        last_height = event.height
        
        # Skip if just moving the window
        if is_just_move:
            return
            
        # Only enforce ratio on actual resize
        desired_width = event.height * aspect_ratio
        desired_height = event.width / aspect_ratio
        
        # Rest of the function remains the same...
    
    # Bind resize event
    root.bind("<Configure>", enforce_aspect_ratio)
    
    # Create client instance
    client = StreamingClient(root)
    
    # Add processor_id property to client
    client.processor_id = 4  # Default to OCR (1)
    
    # Create processor selection dropdown
    processor_frame = tk.Frame(root)
    processor_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    tk.Label(processor_frame, text="Processor Mode:").pack(side='left', padx=5)
    
    processor_options = [
        "Dense Region Caption",
        "OCR",
        "YOLO Detection",
        "MediaPipe",
        "Base Processor",
        "Groq",
        "OpenAI"
    ]
    
    processor_var = tk.StringVar()
    processor_var.set(processor_options[client.processor_id])  # Default to OCR
    
    # Function to update processor_id when dropdown changes
    def on_processor_change(event=None):
        selection = processor_dropdown.current()
        processor_ids = {
            0: 0,  # Dense Region Caption
            1: 1,  # OCR (Default)
            2: 2,  # YOLO
            3: 3,  # MediaPipe
            4: 4,  # Base
            5: 5,  # Groq
            6: 6,  # OpenAI
        }
        client.processor_id = processor_ids.get(selection, 1)
        print(f"Processor changed to: {processor_options[selection]} (ID: {client.processor_id})")
    
    processor_dropdown = ttk.Combobox(processor_frame, 
                                     textvariable=processor_var, 
                                     values=processor_options,
                                     state="readonly",
                                     width=20)
    processor_dropdown.pack(side='left', padx=5)
    processor_dropdown.bind("<<ComboboxSelected>>", on_processor_change)
    on_processor_change()  # Set initial value
    
    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    def safe_stop():
        stop_button.config(state='disabled')
        root.update()  # Update GUI to show button is disabled
        client.stop()
        # Also stop any ongoing TTS
        client.tts_handler.stop_flag.set()
        stop_button.config(state='normal')
    
    start_button = tk.Button(button_frame, text="Start Streaming", command=client.start)
    start_button.pack(side='left', padx=5)
    
    stop_button = tk.Button(button_frame, text="Stop Streaming", command=safe_stop)
    stop_button.pack(side='left', padx=5)
    
    try:
        root.mainloop()
    finally:
        client.stop()

if __name__ == "__main__":
    main()