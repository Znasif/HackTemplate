
def detect_pin_states(self, image, H):
    """
    Detect whether pins are up (reflective) or down at hotspot locations
    using the pixel-based convolution approach adapted for ROIs
    """
    # Create a copy of the input image for visualization
    self.vis_image = image.copy()
    
    # Convert input image to grayscale for brightness analysis
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Get hotspots from model
    hotspots = self.model.get("hotspots", [])
    if not hotspots:
        return False
    
    # Constants for analysis
    PIN_RADIUS = 5  # Radius around pin center to analyze
    b = self.b  # Neighborhood size for convolution (from pixel-based code)
    
    # Extract all positions, rows, and columns at once
    positions = []
    rows = []
    cols = []
    
    for idx, hotspot in enumerate(hotspots):
        if "position" in hotspot:
            positions.append(hotspot["position"])
            count = int(hotspot["hotspotTitle"])
            row, col = int((count-1)/43), (count-1)%43
            rows.append(row)
            cols.append(col)
    
    # Convert to numpy arrays
    positions = np.array(positions, dtype=np.float32)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    if positions.size == 0:
        return False
    
    # Transform all positions to camera view at once
    positions_map = np.array([positions], dtype=np.float32)
    positions_camera = cv.perspectiveTransform(positions_map, np.linalg.inv(H))
    positions_camera = positions_camera[0]
    
    # Convert to integers and filter out points outside the image
    cx_values = positions_camera[:, 0].astype(np.int32)
    cy_values = positions_camera[:, 1].astype(np.int32)
    
    # Create masks for valid points
    valid_mask = (cx_values >= b) & (cx_values < image.shape[1] - b) & \
                (cy_values >= b) & (cy_values < image.shape[0] - b)
    
    # Filter to keep only valid points
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        print("False")
        return False
    
    # Filter arrays to keep only valid points
    cx_values = cx_values[valid_mask]
    cy_values = cy_values[valid_mask]
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    
    # Initialize pin state classifications
    average_response = np.zeros(len(cx_values), dtype=np.float32)
    
    # Process each hotspot's ROI using the pixel-based convolution logic
    for i, (cx, cy, row, col) in enumerate(zip(cx_values, cy_values, rows, cols)):
        # Extract ROI (ensure it's large enough to include the neighborhood)
        x_min = max(0, cx - PIN_RADIUS - b)
        x_max = min(image.shape[1], cx + PIN_RADIUS + b)
        y_min = max(0, cy - PIN_RADIUS - b)
        y_max = min(image.shape[0], cy + PIN_RADIUS + b)
        
        # Get the ROI from the grayscale image
        roi = gray_image[y_min:y_max, x_min:x_max]
        
        # Apply the convolution operation at the center of the ROI
        roi_h, roi_w = roi.shape
        center_x = cx - x_min
        center_y = cy - y_min
        
        # Ensure the center is within bounds for the convolution
        if (center_x >= b and center_x < roi_w - b and 
            center_y >= b and center_y < roi_h - b):
            feature_map = signal.convolve2d(roi, self.kernel, mode='valid')
            # Extract a small region around the center for averaging
            # Adjust for 'valid' mode (reduces size by kernel_size-1)
            fm_y = center_y - b
            fm_x = center_x - b
            # Define a 3x3 region around the center (adjust size as needed)
            region_size = PIN_RADIUS-1
            half_size = region_size // 2
            region_y_min = max(0, fm_y - half_size)
            region_y_max = min(feature_map.shape[0], fm_y + half_size + 1)
            region_x_min = max(0, fm_x - half_size)
            region_x_max = min(feature_map.shape[1], fm_x + half_size + 1)
            
            # Compute the average value in the region
            if (region_y_max > region_y_min and region_x_max > region_x_min):
                region = feature_map[region_y_min:region_y_max, region_x_min:region_x_max]
                average_response[i] = np.mean(region)
            else:
                average_response[i] = 0  # Fallback if region is invalid
    #write code to do k=2 knn based clustering on all the average_response values, it is possible that all the pins may be down in which case the all may belong to down cluster, but it 
    #is highly unlikely to all pins be up, so cluster that way
    classifications = np.zeros(len(cx_values), dtype=np.int32)
    # Apply k-means clustering with k=2 to classify pins as up or down
    if len(average_response) > 0:
        # Reshape for sklearn
        X = average_response.reshape(-1, 1)
        
        # Apply KMeans with k=2
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        cluster_centers = kmeans.cluster_centers_
        
        # Determine which cluster represents "up" pins (higher response values)
        up_cluster = 0 if cluster_centers[0] > cluster_centers[1] else 1
        
        # Classify each pin based on cluster assignment
        classifications = (kmeans.labels_ == up_cluster).astype(np.int32)
        print(np.max(average_response))
        # Special case: If all pins might be down, check the separation between clusters
        # if np.max(average_response) < 1:  # Define appropriate threshold
        #     classifications[:] = 0  # All pins are down
    if np.any(classifications):
        for i, (cx, cy) in enumerate(zip(cx_values, cy_values)):
            if classifications[i]:
                cv.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 255, 0), -1)  # Green for up
            else:
                cv.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 0, 255), -1)  # Red for down
        return True
    return False