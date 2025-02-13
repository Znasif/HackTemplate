class BaseProcessor:
    def __init__(self):
        pass
    
    def process_frame(self, frame):
        """
        Base method for frame processing
        """
        raise NotImplementedError("Subclasses must implement process_frame")