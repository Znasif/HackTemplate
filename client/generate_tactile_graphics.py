import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
import os
import matplotlib.colors as mcolors

class InteractivePinMatrix:
    def __init__(self, pin_matrix=None, block_size=15, title="Tactile Diagram Editor"):
        """
        Initialize an interactive pin matrix editor
        
        Args:
            pin_matrix (numpy.ndarray): Binary 2D array representing pin states (default: empty 31x43)
            block_size (int): Size of each block in pixels
            title (str): Title for the visualization
        """
        self.block_size = block_size
        self.title = title
        
        # Initialize pin matrix if not provided
        if pin_matrix is None:
            self.height, self.width = 31, 43  # Default size
            self.pin_matrix = np.zeros((self.height, self.width), dtype=bool)
        else:
            self.height, self.width = pin_matrix.shape
            self.pin_matrix = pin_matrix.copy()
        
        # Initialize color matrix (r,g,b) for each pin
        self.color_matrix = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Define a palette of 20 colors
        self.colors = self._generate_color_palette(20)
        self.current_color_index = 0
        self.current_color = self.colors[self.current_color_index]
        
        # Keep track of which colors are used in the diagram
        self.used_colors = set()
        self.color_descriptions = {}  # To store descriptions for each color
        
        self.json_data = {
            "title": "New Tactile Diagram",
            "shortDescription": "Interactive tactile diagram",
            "longDescription": "This tactile diagram was created using the interactive pin matrix editor.",
            "creationDate": self._get_current_date(),
            "lastUpdate": self._get_current_date(),
            "hotspots": []
        }
        
        # Create the figure and connect event handlers
        #self._setup_figure()
    
    def _generate_color_palette(self, num_colors):
        """Generate a palette of distinct colors"""
        # Start with some basic colors
        base_colors = [
            [255, 0, 0],      # Red
            [0, 0, 255],      # Blue
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 128, 0],    # Orange
            [128, 0, 255],    # Purple
            [0, 128, 0],      # Dark Green
            [128, 128, 0],    # Olive
            [128, 0, 0],      # Maroon
            [0, 0, 128],      # Navy
            [255, 128, 128],  # Light Red
            [128, 255, 128],  # Light Green
            [128, 128, 255],  # Light Blue
            [192, 192, 192],  # Silver
            [128, 128, 128],  # Gray
            [64, 64, 64],     # Dark Gray
            [255, 215, 0],    # Gold
            [165, 42, 42]     # Brown
        ]
        
        # If more colors are needed, generate them
        if num_colors > len(base_colors):
            # Generate additional colors
            for i in range(len(base_colors), num_colors):
                h = i / num_colors
                # Convert HSV to RGB
                r, g, b = mcolors.hsv_to_rgb([h, 0.8, 0.8])
                base_colors.append([int(r*255), int(g*255), int(b*255)])
        
        return base_colors[:num_colors]
    
    def _get_current_date(self):
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def _setup_figure(self):
        """Set up the matplotlib figure and axes"""
        # Create figure with appropriate size
        self.fig = plt.figure(figsize=(self.width * self.block_size / 80,
                                      self.height * self.block_size / 80 + 2))
        
        # Main axes for pin matrix
        self.ax = plt.subplot2grid((8, 1), (0, 0), rowspan=6)
        
        # Control panel axes
        self.controls_ax = plt.subplot2grid((8, 1), (6, 0), rowspan=2)
        self.controls_ax.axis('off')
        
        # Color indicator
        self.color_indicator_ax = plt.axes([0.85, 0.15, 0.12, 0.06])
        self.color_indicator_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(self.current_color)/255.0))
        self.color_indicator_ax.axis('off')
        self.color_indicator_ax.set_title(f"Color #{self.current_color_index+1}")
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial render
        self._render()
    
    def _update_color_indicator(self):
        """Update the color indicator to show the current color"""
        if hasattr(self, 'color_indicator_ax'):
            self.color_indicator_ax.clear()
            self.color_indicator_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(self.current_color)/255.0))
            self.color_indicator_ax.axis('off')
            self.color_indicator_ax.set_title(f"Color #{self.current_color_index+1}")
            plt.draw()
    
    def _on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        # Convert click coordinates to matrix indices
        i = int(event.ydata // self.block_size)
        j = int(event.xdata // self.block_size)
        
        if 0 <= i < self.height and 0 <= j < self.width:
            # If left click, check if pin already has the same color
            if event.button == 1:
                # If pin is already active and has the same color, deactivate it
                if (self.pin_matrix[i, j] and 
                    np.array_equal(self.color_matrix[i, j], self.current_color)):
                    self.pin_matrix[i, j] = False
                    self.color_matrix[i, j] = [0, 0, 0]
                # Otherwise, set pin to active with current color
                else:
                    self.pin_matrix[i, j] = True
                    self.color_matrix[i, j] = self.current_color
                    # Add current color to used colors
                    color_tuple = tuple(self.current_color)
                    self.used_colors.add(color_tuple)
            elif event.button == 3:
                self.pin_matrix[i, j] = False
                self.color_matrix[i, j] = [0, 0, 0]
            
            # Update visualization
            self._render()
    
    def _on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'up':
            # Cycle to next color
            self.current_color_index = (self.current_color_index + 1) % len(self.colors)
            self.current_color = self.colors[self.current_color_index]
            self._update_color_indicator()
        
        elif event.key == 'down':
            # Cycle to previous color
            self.current_color_index = (self.current_color_index - 1) % len(self.colors)
            self.current_color = self.colors[self.current_color_index]
            self._update_color_indicator()
        
        elif event.key == 'c':
            # Clear the matrix
            self.pin_matrix.fill(False)
            self.color_matrix.fill(0)
            self.used_colors.clear()
            self.color_descriptions.clear()
            self._render()
        
        elif event.key == 't':
            # Change the title
            new_title = input("Enter new title: ")
            self.json_data["title"] = new_title
            short_desc = input("Enter short description: ")
            self.json_data["shortDescription"] = short_desc
            long_desc = input("Enter long description: ")
            self.json_data["longDescription"] = long_desc
            self._render()
        
        elif event.key == 'q':
            # Save and quit - collect descriptions for used colors
            self._gather_color_descriptions()
            self._update_json_hotspots()
            self._save_json(None)
            plt.close(self.fig)
    
    def _gather_color_descriptions(self):
        """Gather descriptions for all used colors"""
        print("\n=== Assign descriptions to colors ===")
        
        for i, color_tuple in enumerate(self.used_colors):
            color = list(color_tuple)
            color_name = self._get_color_name(color)
            
            print(f"\nColor #{i+1}: {color_name} (RGB: {color})")
            
            title = input(f"Enter title for {color_name} areas: ")
            description = input(f"Enter description for {color_name} areas: ")
            sound = input(f"Enter sound file name for {color_name} (or press Enter to skip): ")
            
            # Store in color_descriptions
            self.color_descriptions[color_tuple] = {
                "color": color,
                "hotspotTitle": title,
                "hotspotDescription": description
            }
            
            if sound:
                self.color_descriptions[color_tuple]["sound"] = sound
    
    def _get_color_name(self, color):
        """Get an approximate name for a color"""
        # Define some common colors and their names
        color_names = {
            (255, 0, 0): "Red",
            (0, 255, 0): "Green",
            (0, 0, 255): "Blue",
            (255, 255, 0): "Yellow",
            (255, 0, 255): "Magenta",
            (0, 255, 255): "Cyan",
            (255, 128, 0): "Orange",
            (128, 0, 255): "Purple",
            (0, 128, 0): "Dark Green",
            (128, 128, 0): "Olive",
            (128, 0, 0): "Maroon",
            (0, 0, 128): "Navy",
            (255, 128, 128): "Light Red",
            (128, 255, 128): "Light Green",
            (128, 128, 255): "Light Blue",
            (192, 192, 192): "Silver",
            (128, 128, 128): "Gray",
            (64, 64, 64): "Dark Gray",
            (255, 215, 0): "Gold",
            (165, 42, 42): "Brown"
        }
        
        # Convert to tuple for dictionary lookup
        color_tuple = tuple(color)
        
        # Return name if found, otherwise return "Color #X"
        if color_tuple in color_names:
            return color_names[color_tuple]
        else:
            # Find the closest named color
            min_distance = float('inf')
            closest_name = f"Color #{self.colors.index(color)+1}"
            
            for named_color, name in color_names.items():
                # Calculate Euclidean distance
                distance = sum((c1-c2)**2 for c1, c2 in zip(color_tuple, named_color))
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            
            return closest_name
    
    def _render(self):
        """Update the visualization"""
        self.ax.clear()
        
        # Create image for rendering
        vis_image = np.zeros((self.height * self.block_size, 
                             self.width * self.block_size, 3))
        
        # Fill in the blocks based on pin matrix values and colors
        for i in range(self.height):
            for j in range(self.width):
                if self.pin_matrix[i, j]:
                    color = self.color_matrix[i, j] / 255.0
                    vis_image[i*self.block_size:(i+1)*self.block_size, 
                             j*self.block_size:(j+1)*self.block_size] = color
        
        # Display the image
        self.ax.imshow(vis_image, interpolation='none')
        
        # Add grid lines
        for i in range(self.height + 1):
            self.ax.axhline(i * self.block_size - 0.5, color='gray', linewidth=0.5)
        for j in range(self.width + 1):
            self.ax.axvline(j * self.block_size - 0.5, color='gray', linewidth=0.5)
        
        # Remove axes ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add title
        self.ax.set_title(f"{self.title} ({self.width}x{self.height})")
        
        # Add annotation for controls
        controls_text = "Controls: Left-click to set pins, Right-click to clear pins\n"
        controls_text += "Up/Down arrows to cycle colors, 'c' to clear, 't' to edit title, 'q' to finish and save"
        self.controls_ax.clear()
        self.controls_ax.text(0.5, 0.7, controls_text, 
                             ha='center', va='center',
                             transform=self.controls_ax.transAxes)
        self.controls_ax.axis('off')
        
        plt.tight_layout()
        plt.draw()
    
    def _update_json_hotspots(self):
        """Convert color descriptions to hotspots list for JSON"""
        hotspot_list = []
        
        # Group pins by color
        for color_tuple, description in self.color_descriptions.items():
            # Find all pins with this color
            positions = []
            for i in range(self.height):
                for j in range(self.width):
                    if self.pin_matrix[i, j] and tuple(self.color_matrix[i, j]) == color_tuple:
                        positions.append((i, j))
            
            # Add a hotspot for this color group
            if positions:
                # Use the first position as reference
                hotspot = description.copy()
                hotspot["positions"] = positions  # Add positions to the JSON
                hotspot_list.append(hotspot)
        
        self.json_data["hotspots"] = hotspot_list
        self.json_data["lastUpdate"] = self._get_current_date()
        return self.json_data
    
    def _save_json(self, event):
        """Save the current state as a JSON file"""
        self._update_json_hotspots()
        
        # Get filename from user
        filename = input("Enter filename to save (without .json extension): ")
        if not filename:
            filename = "tactile_diagram"
        
        # Add .json extension if needed
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Save JSON file
        with open(filename, 'w') as f:
            json.dump(self.json_data, f, indent=4)
        
        print(f"JSON saved to {filename}")
    
    def load_from_image(self, image_path):
        """Load pin matrix from an image file"""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        # Convert to pin matrix
        self.pin_matrix = convert_image_to_pin_matrix(img)
        self.height, self.width = self.pin_matrix.shape
        
        # Initialize color matrix with default color (white)
        self.color_matrix = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.height):
            for j in range(self.width):
                if self.pin_matrix[i, j]:
                    self.color_matrix[i, j] = self.current_color
                    # Add default color to used colors
                    self.used_colors.add(tuple(self.current_color))
        
        # Update visualization
        self._setup_figure()
        return True
    
    def load_from_json(self, json_path):
        """Load data from an existing JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Update metadata
            self.json_data.update({
                "title": data.get("title", "Loaded Diagram"),
                "shortDescription": data.get("shortDescription", ""),
                "longDescription": data.get("longDescription", ""),
                "creationDate": data.get("creationDate", self._get_current_date()),
                "lastUpdate": self._get_current_date()
            })
            
            # Reset matrices
            self.pin_matrix = np.zeros((self.height, self.width), dtype=bool)
            self.color_matrix = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.used_colors.clear()
            self.color_descriptions.clear()
            
            # Process hotspots
            for hotspot in data.get("hotspots", []):
                color = tuple(hotspot.get("color", [255, 0, 0]))
                self.used_colors.add(color)
                
                # If positions are specified in the JSON
                if "positions" in hotspot:
                    for i, j in hotspot["positions"]:
                        if 0 <= i < self.height and 0 <= j < self.width:
                            self.pin_matrix[i, j] = True
                            self.color_matrix[i, j] = np.array(color)
                
                # Add to color descriptions
                self.color_descriptions[color] = {
                    "color": list(color),
                    "hotspotTitle": hotspot.get("hotspotTitle", ""),
                    "hotspotDescription": hotspot.get("hotspotDescription", "")
                }
                
                if "sound" in hotspot:
                    self.color_descriptions[color]["sound"] = hotspot["sound"]
            
            # Update visualization
            self._setup_figure()
            return True
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False

# [Rest of the code remains unchanged]
def convert_image_to_pin_matrix(display_image):
    """
    Converts an image to a binary representation for a 43x31 pin matrix,
    emphasizing outlines and key details while suppressing smaller features.

    Args:
        display_image (numpy.ndarray): Input image to be displayed

    Returns:
        numpy.ndarray: Binary 2D array (31x43) representing pin states
    """
    # Convert image to grayscale if it's color
    if len(display_image.shape) == 3:
        grayscale_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = display_image.copy()

    binary_image = cv2.adaptiveThreshold(
        grayscale_image,  # Threshold the resized image
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 1. Apply Gaussian Blur for Noise Reduction and Detail Suppression
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # 2. Apply Edge Detection
    edges = cv2.Canny(blurred_image, 80, 160)

    # 3. Dilate Edges
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 4. Resize *before* thresholding
    resized_edges = cv2.resize(dilated_edges, (43, 31), interpolation=cv2.INTER_AREA)

    # Convert to boolean matrix
    pin_matrix = resized_edges > 0

    return pin_matrix

def create_new_diagram():
    """Create a blank diagram"""
    editor = InteractivePinMatrix()
    plt.show()

def open_from_image(image_path):
    """Create a diagram from an image"""
    editor = InteractivePinMatrix()
    if editor.load_from_image(image_path):
        plt.show()

def open_from_json(json_path):
    """Open an existing diagram JSON file"""
    editor = InteractivePinMatrix()
    if editor.load_from_json(json_path):
        plt.show()

if __name__ == "__main__":
    # Example usage options
    print("Tactile Diagram Editor")
    print("1. Create new blank diagram")
    print("2. Load from image")
    print("3. Load from existing JSON file")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        create_new_diagram()
    elif choice == '2':
        lst = ["plant.jpg", "dolphin.jpg", "amazon_logo.jpg", "flower.jpg", "chart.png", "market_street.png"]
        open_from_image("../server/models/"+lst[-1])
    elif choice == '3':
        json_path = input("Enter JSON file path: ")
        open_from_json(json_path)
    else:
        print("Invalid choice. Starting with blank diagram.")
        create_new_diagram()