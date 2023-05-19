import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageVisualizer:
    def __init__(self, figsize=(15, 5)):
        self.figsize = figsize
        
    def display_images(self, images, layout=None, titles=None, cmap='gray'):
        """
        Display multiple images in a grid.
        
        Parameters:
        - images: List of tuples (title, image)
        - layout: Tuple (rows, cols) for grid layout
        - titles: List of titles for each image
        - cmap: Color map for grayscale images
        """
        n = len(images)
        if layout is None:
            layout = (1, n)
            
        rows, cols = layout
        plt.figure(figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        
        for i, (title, img) in enumerate(images):
            plt.subplot(rows, cols, i+1)
            plt.title(title)
            if len(img.shape) == 2 or cmap is not None:
                plt.imshow(img, cmap=cmap)
            else:
                # Convert BGR to RGB for display
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()