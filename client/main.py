import tkinter as tk
from screen_capture import StreamingClient

def main():
    root = tk.Tk()
    root.title("Screen Capture Client")
    root.geometry("800x600")
    
    client = StreamingClient(root)
    
    button_frame = tk.Frame(root)
    button_frame.pack(side='bottom', fill='x', padx=5, pady=5)
    
    def safe_stop():
        stop_button.config(state='disabled')
        root.update()  # Update GUI to show button is disabled
        client.stop()
        stop_button.config(state='normal')
    
    start_button = tk.Button(button_frame, text="Start Streaming", command=client.start)
    start_button.pack(side='left', padx=5)
    
    stop_button = tk.Button(button_frame, text="Stop Streaming", command=safe_stop)
    stop_button.pack(side='left', padx=5)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        client.stop()
        root.quit()

if __name__ == "__main__":
    main()