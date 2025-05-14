import tkinter as tk
from tkinter import ttk
from screen_capture import StreamingClient
import json
import argparse

# At the top of your code, add these tracking variables
last_x = 0
last_y = 0
last_width = 0
last_height = 0

def main():
    root = tk.Tk()
    root.title("Screen Capture Client")
    root.geometry("853x680")
    
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
    client = StreamingClient(root, monitor_index=1, audio_device_index=5)
    
    # Add processor_id property to client
    client.processor_id = 4  # Default to Base Processor (4)
    
    # Create processor selection dropdown
    processor_frame = tk.Frame(root)
    processor_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    # Create label with a mnemonic (Alt+P will focus the dropdown)
    processor_label = tk.Label(processor_frame, text="Processor Mode (Alt+P):")
    processor_label.pack(side='left', padx=5)
    
    processor_options = [
        "Dense Region Caption",
        "OCR",
        "YOLO Detection",
        "MediaPipe",
        "Base Processor",
        "Groq",
        "OpenAI",
        "Audio Speech-to-Text"
    ]
    
    processor_var = tk.StringVar()
    processor_var.set(processor_options[client.processor_id])  # Default to Base Processor
    
    # Status variable for announcements
    status_var = tk.StringVar()
    status_var.set("Ready")
    
    # Create status label first so we can reference it
    status_frame = tk.Frame(root)
    status_frame.pack(side='bottom', fill='x')
    status_label = tk.Label(status_frame, 
                          textvariable=status_var,
                          bd=1, 
                          relief=tk.SUNKEN, 
                          anchor=tk.W)
    status_label.pack(fill='x')
    status_label.configure(takefocus=1)  # Ensure it can receive focus for screen readers
    
    # Function to update processor_id when dropdown changes
    def on_processor_change(event=None):
        selection = processor_dropdown.current()
        processor_ids = {
            0: 0,  # Dense Region Caption
            1: 1,  # OCR
            2: 2,  # YOLO
            3: 3,  # MediaPipe
            4: 4,  # Base
            5: 5,  # Groq
            6: 6,  # OpenAI
            7: 7,  # Audio Speech-to-Text
        }
        client.processor_id = processor_ids.get(selection, 4)
        selected_processor = processor_options[selection]
        print(f"Processor changed to: {selected_processor} (ID: {client.processor_id})")
        
        # Update status for screen readers
        status_text = f"Selected processor: {selected_processor}"
        status_var.set(status_text)
        
        # Handle special case for audio processor
        if client.processor_id == 7:  # Audio Speech-to-Text
            # Start audio streaming
            client.audio_streamer.start_audio_stream()
            status_text = f"Selected Audio Speech-to-Text - Listening..."
            status_var.set(status_text)
        else:
            # Stop audio streaming if it was running
            client.audio_streamer.stop_audio_stream()
        
        # Set focus to the status label briefly to announce change
        status_label.focus_set()
        root.after(1000, lambda: processor_dropdown.focus_set())
    
    processor_dropdown = ttk.Combobox(processor_frame, 
                                     textvariable=processor_var, 
                                     values=processor_options,
                                     state="readonly",
                                     width=20)
    processor_dropdown.pack(side='left', padx=5)
    processor_dropdown.bind("<<ComboboxSelected>>", on_processor_change)
    
    # Make the dropdown accessible with keyboard
    processor_dropdown.configure(takefocus=1)  # Ensure it can be tabbed to
    
    # Alt+P shortcut for processor dropdown
    def focus_processor(event=None):
        processor_dropdown.focus_set()
        return "break"  # Prevent default handling
        
    root.bind("<Alt-p>", focus_processor)
    
    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    def safe_stop(event=None):
        if stop_button['state'] == 'disabled':
            return
            
        stop_button.config(state='disabled')
        root.update()  # Update GUI to show button is disabled
        client.stop()
        # Also stop any ongoing TTS
        client.tts_handler.stop_flag.set()
        # Make sure audio streaming is stopped
        client.audio_streamer.stop_audio_stream()
        stop_button.config(state='normal')
        
        # Update status for screen readers
        status_text = "Streaming stopped"
        status_var.set(status_text)
        status_label.focus_set()
        root.after(1000, lambda: stop_button.focus_set())
        
        return "break"  # Prevent default handling of the event
    
    def start_streaming(event=None):
        if start_button['state'] == 'disabled':
            return
            
        # Update status for screen readers
        status_text = "Streaming started"
        status_var.set(status_text)
        
        # Start the appropriate type of streaming based on processor
        if client.processor_id == 7:  # Audio Speech-to-Text
            # For audio processing, we only need to start the audio streamer
            client.audio_streamer.start_audio_stream()
            status_text = "Audio streaming started - Listening..."
            status_var.set(status_text)
        else:
            # For video processing, start the regular streaming
            client.start()
        
        # Set focus to status label briefly to announce change
        status_label.focus_set()
        root.after(1000, lambda: start_button.focus_set())
        
        return "break"  # Prevent default handling of the event
    
    # Create buttons with mnemonics
    start_button = tk.Button(button_frame, text="Start Streaming (Alt+S)", command=start_streaming)
    start_button.pack(side='left', padx=5)
    
    stop_button = tk.Button(button_frame, text="Stop Streaming (Alt+X)", command=safe_stop)
    stop_button.pack(side='left', padx=5)
    
    # Add keyboard shortcuts using Alt keys (more accessible than Ctrl)
    root.bind("<Alt-s>", start_streaming)
    root.bind("<Alt-x>", safe_stop)
    
    # Call on_processor_change to set initial value
    on_processor_change()
    
    # Add help dialog
    def show_help(event=None):
        help_text = """
        Keyboard Controls:
        - Tab: Navigate between controls
        - Alt+S: Start streaming
        - Alt+X: Stop streaming
        - Alt+P: Focus processor dropdown
        - Alt+H: Show this help dialog
        - Alt+Q: Focus quality slider
        - Alt+L: Focus left cropping control
        - Alt+R: Focus right cropping control
        - Alt+T: Focus top cropping control
        - Alt+B: Focus bottom cropping control
        - Arrow keys: Adjust values when a control is focused
        - Escape: Close this dialog
        
        When focused on the processor dropdown:
        - Use Up/Down arrows to navigate options
        - Press Enter to select an option
        """
        help_window = tk.Toplevel(root)
        help_window.title("Keyboard Accessibility Help")
        help_window.geometry("400x300")
        help_window.resizable(False, False)
        
        help_label = tk.Label(help_window, text=help_text, justify=tk.LEFT, padx=20, pady=20)
        help_label.pack(fill=tk.BOTH, expand=True)
        
        # Make help window keyboard accessible
        help_window.grab_set()  # Make modal
        help_window.focus_set()
        
        # Close on Escape key
        help_window.bind("<Escape>", lambda e: help_window.destroy())
        
        # Close button with keyboard accessibility
        close_button = tk.Button(help_window, text="Close (Escape)", command=help_window.destroy)
        close_button.pack(pady=10)
        close_button.focus_set()  # Set initial focus to close button
        
        return "break"  # Prevent default handling
    
    # Add help button
    help_button = tk.Button(button_frame, text="Help (Alt+H)", command=show_help)
    help_button.pack(side='right', padx=5)
    
    # Add keyboard shortcut for help
    root.bind("<Alt-h>", show_help)

        # Focus quality slider
    def focus_quality(event=None):
        client.quality_slider.focus_set()
        return "break"
        
    root.bind("<Alt-q>", focus_quality)

    # Focus cropping controls
    def focus_left_pane(event=None):
        client.left_pane.focus_set()
        return "break"
        
    def focus_right_pane(event=None):
        client.right_pane.focus_set()
        return "break"
        
    def focus_top_pane(event=None):
        client.top_pane.focus_set()
        return "break"
        
    def focus_bottom_pane(event=None):
        client.bottom_pane.focus_set()
        return "break"
        
    root.bind("<Alt-l>", focus_left_pane)
    root.bind("<Alt-r>", focus_right_pane)
    root.bind("<Alt-t>", focus_top_pane)
    root.bind("<Alt-b>", focus_bottom_pane)
    
    # Set initial focus
    start_button.focus_set()
    
    # Override client methods to update status
    original_start = client.start
    def accessible_start():
        original_start()
        status_text = "Streaming started"
        status_var.set(status_text)
    client.start = accessible_start
    
    original_stop = client.stop
    def accessible_stop():
        original_stop()
        client.audio_streamer.stop_audio_stream()  # Always ensure audio stream is stopped
        status_text = "Streaming stopped"
        status_var.set(status_text)
    client.stop = accessible_stop
    
    try:
        root.mainloop()
    finally:
        client.stop()

if __name__ == "__main__":
    main()