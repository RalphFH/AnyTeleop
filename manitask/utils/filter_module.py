import numpy as np
import collections


class Filter:
    """
    A simple filter class for smoothing joint trajectories.
    Provides multiple filtering methods and outlier detection.
    """
    
    def __init__(self, filter_type="moving_average", window_size=3, alpha=0.3, jump_threshold=0.1):
        """
        Initialize the filter
        
        Args:
            filter_type (str): Filter type, options: "moving_average", "exponential", "median", "none"
            window_size (int): Moving window size for moving average and median filters
            alpha (float): Smoothing factor for exponential filter (0-1), higher means more smoothing
            jump_threshold (float): Threshold for outlier detection
        """
        self.filter_type = filter_type
        self.window_size = window_size
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        
        # Internal state
        self.point_history = collections.deque(maxlen=window_size)
        self.last_points = None
    
    def filter(self, points):
        """        
        Args:
            points (np.ndarray): Array of point positions, shape (n_points, 3)
            is_detected (bool): Whether hand is detected
            
        Returns:
            tuple: (is_valid, filtered_points)
                is_valid (bool): Whether valid filtering result exists
                filtered_points (np.ndarray): Filtered point positions
        """
                
        # If no filtering is needed
        if self.filter_type == "none":
            self.last_points = points.copy()
            return True, points
        
        # Outlier detection
        # is_outlier = False
        # if self.last_points is not None:
        #     # Calculate maximum displacement of joints
        #     max_displacement = np.max(np.linalg.norm(points - self.last_points, axis=1))
        #     is_outlier = max_displacement > self.jump_threshold
        
        # if is_outlier and self.last_points is not None:
        #     # If outlier detected, keep using previous frame
        #     return True, self.last_points
        
        if self.filter_type == "moving_average":
            # Moving average filter
            self.point_history.append(points)
            if len(self.point_history) > 0:
                filtered_points = np.mean(self.point_history, axis=0)
            else:
                filtered_points = points
                
        elif self.filter_type == "exponential":
            # Exponential smoothing filter
            if self.last_points is not None:
                filtered_points = self.alpha * points + (1 - self.alpha) * self.last_points
            else:
                filtered_points = points
                
        elif self.filter_type == "median":
            # Median filter
            self.point_history.append(points)
            if len(self.point_history) > 2:  # Need at least 3 samples
                filtered_points = np.median(self.point_history, axis=0)
            else:
                filtered_points = points
        
        else:
            # Default: no filtering
            filtered_points = points
        
        # Update previous frame data
        self.last_points = filtered_points.copy()
        return True, filtered_points
    
    def reset(self):
        """Reset filter state"""
        self.point_history.clear()
        self.last_points = None
        self.missed_frames = 0
    
    def set_params(self, filter_type=None, window_size=None, alpha=None, jump_threshold=None, max_missed_frames=None):
        """Update filter parameters"""
        if filter_type is not None:
            self.filter_type = filter_type
            
        if window_size is not None and window_size != self.window_size:
            # Need to recreate queue when window size changes
            old_data = list(self.point_history)
            self.window_size = window_size
            self.point_history = collections.deque(maxlen=window_size)
            # Keep most recent data
            for item in old_data[-window_size:]:
                self.point_history.append(item)
                
        if alpha is not None:
            self.alpha = alpha
            
        if jump_threshold is not None:
            self.jump_threshold = jump_threshold
            
