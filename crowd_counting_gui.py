import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import matplotlib.backends.backend_qt5agg as plt_backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import torchvision.transforms.functional as F
from torchvision import transforms
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFileDialog, QWidget, QSplitter, 
                             QFrame, QSizePolicy, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

# Import the CSRNet model
# This imports your existing model implementation
from model import CSRNet

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class CrowdCountingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CSRNet Crowd Counting Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize model variable
        self.model = None
        
        # Define the transformation for image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize UI first to set up model selector
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Top section: Instructions
        instruction_label = QLabel("Upload an image to count the number of people")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(instruction_label)
        
        # Middle section: Image display, heatmap display, and controls
        middle_section = QSplitter(Qt.Horizontal)
        
        # Left panel for original image
        left_panel = QFrame()
        left_layout = QVBoxLayout()
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.image_label.setMinimumSize(400, 400)
        left_layout.addWidget(self.image_label)
        left_panel.setLayout(left_layout)
        
        # Right panel for density map
        right_panel = QFrame()
        right_layout = QVBoxLayout()
        self.density_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.density_canvas)
        self.count_label = QLabel("Predicted count: -")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.count_label)
        right_panel.setLayout(right_layout)
        
        # Add panels to middle section
        middle_section.addWidget(left_panel)
        middle_section.addWidget(right_panel)
        
        # Add middle section to main layout
        main_layout.addWidget(middle_section)
        
        # Bottom section: Buttons
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setStyleSheet("font-size: 14px;")
        
        self.process_btn = QPushButton("Count People")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("font-size: 14px;")
        self.process_btn.setEnabled(False)
        
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.process_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid lightgray;")
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Top section: Instructions and model selection
        top_layout = QHBoxLayout()
        
        instruction_label = QLabel("Upload an image to count the number of people")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        top_layout.addWidget(instruction_label, 3)
        
        # Model selection dropdown
        model_layout = QHBoxLayout()
        model_label = QLabel("Select model:")
        model_label.setStyleSheet("font-size: 14px;")
        self.model_selector = QComboBox()
        self.model_selector.addItem("Part A model", "PartAmodel_best.pth.tar")
        self.model_selector.addItem("Part B model", "partBmodel_best.pth.tar")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        top_layout.addLayout(model_layout, 1)
        
        main_layout.addLayout(top_layout)
        
        # Middle section: Image display, heatmap display, and controls
        middle_section = QSplitter(Qt.Horizontal)
        
        # Left panel for original image
        left_panel = QFrame()
        left_layout = QVBoxLayout()
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.image_label.setMinimumSize(400, 400)
        left_layout.addWidget(self.image_label)
        left_panel.setLayout(left_layout)
        
        # Right panel for density map
        right_panel = QFrame()
        right_layout = QVBoxLayout()
        self.density_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.density_canvas)
        self.count_label = QLabel("Predicted count: -")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.count_label)
        right_panel.setLayout(right_layout)
        
        # Add panels to middle section
        middle_section.addWidget(left_panel)
        middle_section.addWidget(right_panel)
        
        # Add middle section to main layout
        main_layout.addWidget(middle_section)
        
        # Bottom section: Buttons
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setStyleSheet("font-size: 14px;")
        
        self.process_btn = QPushButton("Count People")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("font-size: 14px;")
        self.process_btn.setEnabled(False)
        
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.process_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid lightgray;")
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def load_model(self):
        try:
            self.status_label.setText("Loading model...")
            QApplication.processEvents()
            
            # Get the selected model file
            model_file = self.model_selector.currentData()
            
            self.model = CSRNet()
            
            # Load pre-trained weights
            checkpoint = torch.load(model_file, 
                                   map_location=torch.device('cpu'),
                                   weights_only=False)
            
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            self.status_label.setText(f"Model {model_file} loaded successfully!")
        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
    
    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", 
                                                 options=options)
        
        if file_path:
            try:
                self.image_path = file_path
                self.display_image(file_path)
                self.process_btn.setEnabled(True)
                self.status_label.setText(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.setText(f"Error loading image: {str(e)}")
    
    def display_image(self, image_path):
        # Display the original image
        pixmap = QPixmap(image_path)
        
        # Scale pixmap to fit in the label while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
    
    def process_image(self):
        # Load the model if not already loaded or if model selection changed
        self.load_model()
        
        if not hasattr(self, 'image_path'):
            self.status_label.setText("No image selected.")
            return
        
        try:
            self.status_label.setText("Processing image...")
            QApplication.processEvents()
            
            # Load and preprocess the image using the transform method
            img = Image.open(self.image_path).convert('RGB')
            transformed_img = self.transform(img).unsqueeze(0)
            
            # Forward pass through the model
            with torch.no_grad():
                output = self.model(transformed_img)
            
            # Get the predicted count
            predicted_count = int(output.sum().item())
            
            # Display the density map
            density_map = output.squeeze().detach().cpu().numpy()
            
            # Clear the previous plot
            self.density_canvas.axes.clear()
            
            # Plot the density map
            im = self.density_canvas.axes.imshow(density_map, cmap=CM.jet)
            self.density_canvas.axes.set_title("Density Map")
            self.density_canvas.axes.axis('off')
            self.density_canvas.figure.colorbar(im)
            self.density_canvas.draw()
            
            # Update the count label
            self.count_label.setText(f"Predicted count: {predicted_count}")
            
            # Also display the original image with the count
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(density_map, cmap=CM.jet)
            plt.title(f"Density Map (Count: {predicted_count})")
            plt.axis('off')
            plt.colorbar()
            
            self.status_label.setText("Processing complete!")
            
        except Exception as e:
            self.status_label.setText(f"Error processing image: {str(e)}")
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set stylesheet for a nicer appearance
    app.setStyle("Fusion")
    
    # Create and show the application window
    window = CrowdCountingGUI()
    window.show()
    
    # Display usage instructions
    print("==== CSRNet Crowd Counting Application ====")
    print("1. Select which model to use (Part A or Part B)")
    print("2. Click 'Upload Image' to select an image")
    print("3. Click 'Count People' to process the image")
    print("4. The application will display the density map and predicted count")
    print("===========================================")
    
    sys.exit(app.exec_())