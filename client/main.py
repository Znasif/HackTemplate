import tkinter as tk
from screen_capture import StreamingClient

def main():
    root = tk.Tk()
    root.title("Screen Capture Client")
    root.geometry("800x600")  # Set initial window size
    
    client = StreamingClient(root)
    
    # Create frame for buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    # Add buttons
    start_button = tk.Button(button_frame, text="Start Streaming", command=client.start)
    start_button.pack(side='left', padx=5)
    
    stop_button = tk.Button(button_frame, text="Stop Streaming", command=client.stop)
    stop_button.pack(side='left', padx=5)
    
    # Start the tkinter main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        client.stop()
        root.quit()

if __name__ == "__main__":
    main()