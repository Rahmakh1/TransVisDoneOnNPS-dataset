import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog,QProgressBar,QHBoxLayout,QSizePolicy,QSlider,QComboBox,QButtonGroup,QRadioButton,QLineEdit,QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint,QRect
from PyQt5.QtGui import QImage, QPixmap
from utilsGUI import load_model, letterbox_temporal, non_max_suppression,apply_gamma_correction,apply_histogram_equalization
import os
import time # Add this import at the top of the file

#enable high-DPI support
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"  # Enable automatic scaling
os.environ["QT_SCALE_FACTOR"] = "1"  # Set a fixed scale factor (optional)


class InferenceThread(QThread):

    """A thread to handle frame-by-frame inference and visualization."""

    frame_ready = pyqtSignal(np.ndarray)  # Signal to emit processed frames
    progress_update = pyqtSignal(int)  # Signal for progress updates
    fps_update = pyqtSignal(float)  # New signal to emit FPS updates

    def __init__(self, video_path, model, device, img_size=640, num_frames=5, stride=32,
                 conf_thres=0.3, iou_thres=0.6, augment=False, single_cls=False,gamma=1.0,histogram_equalization=False):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.device = device
        self.img_size = img_size
        self.num_frames = num_frames
        self.stride = stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.single_cls = single_cls
        self.running = True
        self.gamma = gamma  # Gamma value for degradation
        self.histogram_equalization = histogram_equalization  # Histogram equalization flag

    def stop(self):
        self.running = False
        self.wait()

    def run(self):

        """Process frames one by one and emit them with predictions."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Failed to open video file: {self.video_path}")
            return

        # Get total frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        frames = []
        start_time = time.time()  # Start time for FPS calculation

        while self.running:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
            # Apply histogram equalization if enabled
            if self.histogram_equalization:
                frame = apply_histogram_equalization(frame)

            # Apply gamma correction if gamma != 1
            if self.gamma != 1.0:
                frame = apply_gamma_correction(frame, self.gamma)

            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # Check if we have enough frames to form a clip
            if len(frames) == self.num_frames:
                # Preprocess the frames
                preprocessed_frames = []
                resize_info = []
                for frame in frames:
                    resized_frame, ratio, (dw, dh) = letterbox_temporal([frame], new_shape=self.img_size, stride=self.stride)
                    resized_frame = resized_frame[0]  # Unpack single frame
                    resize_info.append((ratio, (dw, dh)))

                    # Normalize pixel values to [0, 1]
                    normalized_frame = resized_frame.astype(np.float32) / 255.0

                    # Convert HWC to CHW (Height, Width, Channels -> Channels, Height, Width)
                    chw_frame = np.transpose(normalized_frame, (2, 0, 1))

                    preprocessed_frames.append(chw_frame)

                # Stack frames into a clip (T, C, H, W)
                clip = np.stack(preprocessed_frames, axis=0)

                # Convert clip to a PyTorch tensor
                clip = torch.from_numpy(clip)

                # Move clip to device and cast to correct precision
                half = self.device.type != 'cpu'  # Half precision only supported on CUDA
                clip = clip.to(self.device, non_blocking=True)
                clip = clip.half() if half else clip.float()

                # Run model
                with torch.no_grad():
                    out, _ = self.model(clip, augment=self.augment)

                # Apply Non-Maximum Suppression
                predictions = non_max_suppression(out, self.conf_thres, self.iou_thres, classes=0, agnostic=self.single_cls)

                # Visualize predictions on the oldest frame
                orig_frame = frames[0]
                orig_h, orig_w = orig_frame.shape[:2]
                if predictions and predictions[0] is not None:  # Check if predictions exist
                    det = predictions[0]  # Predictions for the oldest frame
                    det = det.cpu().numpy()
                    for box in det:
                        x1, y1, x2, y2, conf, cls = box[:6]

                        # Get resize and padding info for this frame
                        (scale_x, scale_y), (pad_x, pad_y) = resize_info[0]

                        # Scale bounding box coordinates back to original resolution
                        try:
                            x1 = int((x1 - pad_x) / scale_x)
                            y1 = int((y1 - pad_y) / scale_y)
                            x2 = int((x2 - pad_x) / scale_x)
                            y2 = int((y2 - pad_y) / scale_y)
                        except (ZeroDivisionError, OverflowError) as e:
                            print(f"Error scaling bounding box: {e}")
                            continue

                        # Ensure coordinates are within bounds
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)

                        # Draw bounding box
                        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add label
                        label = f"Class {int(cls)}, Conf {conf:.2f}"
                        cv2.putText(orig_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Emit the processed frame
                self.frame_ready.emit(cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR))

                # Remove the oldest frame from the buffer
                frames = frames[1:]

            # Update progress
            processed_frames += 1
            progress = int((processed_frames / total_frames) * 100)
            self.progress_update.emit(progress)  # Emit progress update

            # Calculate and emit FPS
            elapsed_time = time.time() - start_time
            fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
            self.fps_update.emit(fps)  # Emit FPS update

        cap.release()

class VideoInferenceApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window properties
        self.setWindowTitle("TransVisDrone Inference App")
        self.setGeometry(100, 100, 1200, 800)  # Increased width for better layout

        # Variables for confidence threshold
        self.confidence_threshold = 0.1  # Default confidence threshold

        # Variables for zoom functionality
        self.zoom_factor = 1.0  # Initial zoom factor (1.0 = no zoom)
        self.zoom_point = QPoint()  # Point to zoom around (mouse position)
        self.current_frame = None  # Store the current frame for zooming
        self.visible_region = QRect()  # Tracks the visible region of the frame

        # Main widget and layout (horizontal layout)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)  # Horizontal layout

        # Sidebar widget (vertical layout)
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)  # Vertical layout
        self.sidebar_widget.setFixedWidth(300)  # Fixed width for the sidebar
        self.sidebar_layout.setSpacing(10)  # Space between buttons
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)  # Margins around the sidebar
        self.main_layout.addWidget(self.sidebar_widget)  # Add sidebar to the main layout

        # Video display widget
        self.video_widget = QWidget()
        self.video_layout = QVBoxLayout(self.video_widget)  # Vertical layout for video display
        self.video_layout.setSpacing(10)
        self.video_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.addWidget(self.video_widget)  # Add video widget to the main layout

        # Label for displaying video frames
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout.addWidget(self.video_label)

        # Connect mouse wheel event
        self.video_label.wheelEvent = self.wheel_event

        # Progress Bar and FPS Layout (Horizontal layout)
        self.progress_fps_layout = QHBoxLayout()  # Horizontal layout for progress bar, confidence, and FPS
        self.video_layout.addLayout(self.progress_fps_layout)  # Add to the video layout

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.video_layout.addWidget(self.progress_bar)  # Add progress bar to the video layout



        
 
        # Load the model when the application starts
        print("Loading model...")
        self.model, self.device = load_model()  # Load the model here

        # Buttons (added to the sidebar)
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.sidebar_layout.addWidget(self.load_button)

        # Confidence Threshold Slider
        self.conf_slider_label = QLabel("Set Confidence Threshold: 10% ", self)
        self.sidebar_layout.addWidget(self.conf_slider_label)

        self.conf_slider = QSlider(Qt.Horizontal, self)
        self.conf_slider.setMinimum(1)  # Minimum value: 1% (0.01)
        self.conf_slider.setMaximum(100)  # Maximum value: 100% (1.0)
        self.conf_slider.setValue(10)  # Default value: 10% (0.1)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.sidebar_layout.addWidget(self.conf_slider)

        # Degradation Section
        self.degradation_label = QLabel(" Brightness Degradation:", self)
        self.sidebar_layout.addWidget(self.degradation_label)

        # Histogram Equalization Checkbox
        self.histogram_equalization_checkbox = QCheckBox("Apply Histogram Equalization", self)
        self.sidebar_layout.addWidget(self.histogram_equalization_checkbox)

        # Radio buttons for degradation options
        self.degradation_group = QButtonGroup(self)
        self.no_degradation_radio = QRadioButton("without Gamma function")
        self.brightening_radio = QRadioButton("Brightening with Gamma")
        self.darkening_radio = QRadioButton("Darkening with Gamma")

        # Add radio buttons to the group and layout
        self.degradation_group.addButton(self.no_degradation_radio)
        self.degradation_group.addButton(self.brightening_radio)
        self.degradation_group.addButton(self.darkening_radio)
        self.sidebar_layout.addWidget(self.no_degradation_radio)
        self.sidebar_layout.addWidget(self.brightening_radio)
        self.sidebar_layout.addWidget(self.darkening_radio)

        # Set default selection
        self.no_degradation_radio.setChecked(True)

        # Gamma value input field
        self.gamma_input = QLineEdit(self)
        self.gamma_input.setPlaceholderText("Enter gamma value...")
        self.gamma_input.setVisible(False)  # Initially hidden
        self.sidebar_layout.addWidget(self.gamma_input)

        # Error message label
        self.gamma_error_label = QLabel("", self)
        self.gamma_error_label.setStyleSheet("color: red;")
        self.gamma_error_label.setVisible(False)
        self.sidebar_layout.addWidget(self.gamma_error_label)

        # Connect radio buttons to toggle gamma input visibility
        self.no_degradation_radio.toggled.connect(lambda: self.toggle_gamma_input(False))
        self.brightening_radio.toggled.connect(lambda: self.toggle_gamma_input(True))
        self.darkening_radio.toggled.connect(lambda: self.toggle_gamma_input(True))

        # Variables for degradation
        self.gamma_value = 1.0  # Default gamma value


        self.inference_button = QPushButton("Start Inference")
        self.inference_button.clicked.connect(self.start_inference)
        self.inference_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.inference_button)

        self.stop_button = QPushButton("Stop Inference")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.stop_button)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.reset_zoom_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.reset_zoom_button)

        # Add stretch to push buttons to the top of the sidebar
        self.sidebar_layout.addStretch()

        # Save Processed Video Button
        self.save_button = QPushButton("Save Processed Video")
        self.save_button.clicked.connect(self.save_processed_video)
        self.save_button.setEnabled(False)  # Initially disabled
        self.sidebar_layout.addWidget(self.save_button)

        

        # Variables for saving the processed video
        self.processed_frames = []  # List to store processed frames during inference
        self.fps = None  # Frame rate of the original video
        self.frame_size = None  # Resolution of the original video

        # Confidence Threshold Display Label
        self.conf_display_label = QLabel("", self)
        self.conf_display_label.setVisible(False)  # Initially hidden
        self.progress_fps_layout.addWidget(self.conf_display_label)  # Add confidence label to the horizontal layout

        # Add a stretch to push the FPS label to the right
        self.progress_fps_layout.addStretch()

        # FPS Display Label
        self.fps_label = QLabel("FPS: 0.00", self)
        self.fps_label.setVisible(False)  # Initially hidden
        self.progress_fps_layout.addWidget(self.fps_label)  # Add FPS label to the horizontal layout

        # Initialize inference thread as None
        self.inference_thread = None

    def resizeEvent(self, event):
        """
        Handle window resize events.
        Adjust the sidebar width to be 1/5 of the total window width.
        """
        # Call the base class implementation
        super().resizeEvent(event)

        # Update the sidebar width
        new_sidebar_width = int(self.width() / 5)
        self.sidebar_widget.setFixedWidth(new_sidebar_width)

    def reset_zoom(self):
        """Reset the zoom to the default view."""
        self.zoom_factor = 1.0  # Reset zoom factor
        self.visible_region = QRect()  # Reset visible region
        self.update_frame(self.current_frame)  # Redisplay the frame
        self.reset_zoom_button.setEnabled(False)  # Disable reset button

    def update_conf_threshold(self, value):
        """Update the confidence threshold based on the slider value."""
        self.confidence_threshold = value / 100.0  # Convert percentage to decimal
        self.conf_slider_label.setText(f"Confidence Threshold: {value}%")

    def wheel_event(self, event):
        """Handle mouse wheel event to zoom in/out."""
        # Determine zoom direction
        delta = event.angleDelta().y()  # Get scroll amount
        if delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        elif delta < 0:
            self.zoom_factor *= 0.9  # Zoom out

        # Clamp zoom factor to reasonable limits
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))

        # Store the mouse position for zooming around it
        self.zoom_point = event.pos()

        # Calculate the new visible region
        self.update_visible_region()

        # Update the frame display
        self.update_frame(self.current_frame)
        self.reset_zoom_button.setEnabled(self.zoom_factor != 1.0)  # Enable reset button if zoomed
        def reset_zoom(self):
            """Reset the zoom to the default view."""
            self.zoom_factor = 1.0  # Reset zoom factor
            self.update_frame(self.current_frame)  # Redisplay the frame
            self.reset_zoom_button.setEnabled(False)  # Disable reset button

    def update_visible_region(self):
        """Update the visible region of the frame based on the zoom factor and mouse position."""
        if self.current_frame is None:
            return

        # Get the dimensions of the video label
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # Calculate the zoomed frame size
        zoomed_width = int(self.current_frame.shape[1] * self.zoom_factor)
        zoomed_height = int(self.current_frame.shape[0] * self.zoom_factor)

        # Map the mouse position to the zoomed frame's coordinate system
        zoom_x = int(self.zoom_point.x() * self.zoom_factor)
        zoom_y = int(self.zoom_point.y() * self.zoom_factor)

        # Calculate the top-left corner of the visible region
        x = zoom_x - label_width // 2
        y = zoom_y - label_height // 2

        # Ensure the visible region stays within the zoomed frame bounds
        x = max(0, min(x, zoomed_width - label_width))
        y = max(0, min(y, zoomed_height - label_height))

        # Update the visible region
        self.visible_region = QRect(x, y, label_width, label_height)

    def update_frame(self, frame):
        """Update the video label with the processed frame, applying zoom."""
        if frame is None:
            return

        self.current_frame = frame  # Store the current frame

        # Get the dimensions of the video label
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # Calculate the zoomed frame size
        zoomed_width = int(frame.shape[1] * self.zoom_factor)
        zoomed_height = int(frame.shape[0] * self.zoom_factor)

        # Resize the frame based on the zoom factor
        zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))

        # Crop the zoomed frame to the visible region
        if self.zoom_factor > 1.0:
            x, y = self.visible_region.x(), self.visible_region.y()
            width, height = self.visible_region.width(), self.visible_region.height()
            zoomed_frame = zoomed_frame[y:y + height, x:x + width]
        else:
            # If zoomed out, resize the frame to fit the label
            zoomed_frame = cv2.resize(zoomed_frame, (label_width, label_height))

        # Convert the frame from BGR (OpenCV default) to RGB
        zoomed_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)

        # Get frame dimensions
        height, width, channel = zoomed_frame.shape
        bytes_per_line = 3 * width

        # Create QImage using RGB format
        q_img = QImage(zoomed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display the frame in the label
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def load_video(self):
        """Open a file dialog to select a video file."""
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mov *.mp4 *.avi);;All Files (*)",
            options=options
        )
        if self.video_path:
            self.inference_button.setEnabled(True)

    def toggle_gamma_input(self, visible):
        """Toggle visibility of the gamma input field."""
        self.gamma_input.setVisible(visible)
        self.gamma_error_label.setVisible(False)  # Clear error message

    def validate_gamma_value(self):
        """Validate the gamma value entered by the user."""
        try:
            gamma = float(self.gamma_input.text())
            if self.brightening_radio.isChecked():
                if 0 < gamma < 1:
                    self.gamma_value = gamma
                    self.gamma_error_label.setVisible(False)
                    return True
                else:
                    self.gamma_error_label.setText("Error: Brightening requires 0 < γ < 1.")
                    self.gamma_error_label.setVisible(True)
                    return False
            elif self.darkening_radio.isChecked():
                if gamma > 1:
                    self.gamma_value = gamma
                    self.gamma_error_label.setVisible(False)
                    return True
                else:
                    self.gamma_error_label.setText("Error: Darkening requires γ > 1.")
                    self.gamma_error_label.setVisible(True)
                    return False
        except ValueError:
            self.gamma_error_label.setText("Error: Invalid gamma value.")
            self.gamma_error_label.setVisible(True)
            return False

    def start_inference(self):
        """Start the inference thread."""

        # Validate gamma value if degradation is enabled
        if not self.no_degradation_radio.isChecked():
            if not self.validate_gamma_value():
                print("Invalid gamma value. Cannot start inference.")
                return

        # Disable degradation section during inference
        self.no_degradation_radio.setEnabled(False)
        self.brightening_radio.setEnabled(False)
        self.darkening_radio.setEnabled(False)
        self.gamma_input.setEnabled(False)
        self.histogram_equalization_checkbox.setEnabled(False)
        
        if not self.video_path:
            return

        # Reset variables for saving the processed video
        self.processed_frames = []
        self.fps = None
        self.frame_size = None

        # Reset FPS tracking
        self.fps_values = []

        


        # Disable buttons during inference
        self.load_button.setEnabled(False)
        self.inference_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)

        # Show the progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Hide the confidence slider and label
        self.conf_slider_label.setVisible(False)
        self.conf_slider.setVisible(False)

        # Display the selected confidence threshold under the progress bar
        self.conf_display_label.setText(f"Confidence Threshold: {int(self.confidence_threshold * 100)}%")
        self.conf_display_label.setVisible(True)

        # Show the FPS label
        self.fps_label.setVisible(True)

        # Start the inference thread with the preloaded model and selected confidence threshold
        self.inference_thread = InferenceThread(
            self.video_path, self.model, self.device, conf_thres=self.confidence_threshold,gamma=self.gamma_value
        )
        self.inference_thread.frame_ready.connect(self.update_frame_and_save)
        self.inference_thread.progress_update.connect(self.update_progress)  # Connect progress signal
        self.inference_thread.fps_update.connect(self.update_fps)  # Connect FPS signal
        self.inference_thread.finished.connect(self.inference_finished)  # Connect finished signal
        self.inference_thread.start()

    def update_fps(self, fps):

        """Update the FPS label."""

        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.fps_values.append(fps)  # Store FPS value for later averaging

    def update_frame_and_save(self, frame):
        """Update the video label and save the processed frame."""
        # Update the video display
        self.update_frame(frame)
        

        # Store the processed frame for saving later
        if self.fps is None or self.frame_size is None:
            cap = cv2.VideoCapture(self.video_path)
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate
            self.frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            cap.release()
        self.processed_frames.append(frame)

    def save_processed_video(self):
        """Save the processed video with predictions overlaid."""

        if not self.processed_frames:
            print("No processed frames to save.")
            return

        # Open a file dialog to select the output file path
        options = QFileDialog.Options()
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Video",
            "",
            "Video Files (*.mp4);;All Files (*)",
            options=options,
        )
        if not output_path:
            return

        # Ensure the output path has the correct extension
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        # Write each processed frame to the video
        for frame in self.processed_frames:
            # Resize the frame to match the original video resolution
            resized_frame = cv2.resize(frame, self.frame_size)
            out.write(resized_frame)

        # Release the VideoWriter
        out.release()
        print(f"Processed video saved to: {output_path}")



    
    def update_progress(self, progress):
        """Update the progress bar."""
        self.progress_bar.setValue(progress)

    def inference_finished(self):
        """
        Handle the completion of the inference thread.
        Re-enable buttons and hide the progress bar.

        """
        
        self.progress_bar.setValue(0)  # Reset progress bar to 0%

        # Re-enable buttons
        self.load_button.setEnabled(True)
        self.inference_button.setEnabled(True)
        self.save_button.setEnabled(True)  # Enable the save button
        self.stop_button.setEnabled(False)

        # Re-enable the degradation section
        self.no_degradation_radio.setEnabled(True)
        self.brightening_radio.setEnabled(True)
        self.darkening_radio.setEnabled(True)
        self.gamma_input.setEnabled(True)
        self.histogram_equalization_checkbox.setEnabled(True)

        # Show the slider and its label again
        self.conf_slider_label.setVisible(True)
        self.conf_slider.setVisible(True)
        
        # Hide the confidence threshold display label
        self.conf_display_label.setVisible(False)
        
        # Hide the real time FPS label
        self.fps_label.setVisible(False)

        # Reset degradation settings
        self.gamma_input.setText("")  # Clear gamma input field
        self.gamma_value = 1.0  # Reset gamma value to default
        self.no_degradation_radio.setChecked(True)  # Select "No Degradation" by default
        self.histogram_equalization_checkbox.setChecked(False)  # Uncheck histogram equalization

        

        print("Inference completed.")


    def stop_inference(self):
        """Stop the inference thread."""
        if self.inference_thread and self.inference_thread.isRunning():
            print("Stopping inference thread...")
            self.inference_thread.stop()
            self.inference_finished()  # Clean up UI
    
    def closeEvent(self, event):
        """
        Handle the application close event.
        Ensures the inference thread is stopped gracefully before closing.
        """
        if self.inference_thread and self.inference_thread.isRunning():
            print("Stopping inference thread...")
            self.inference_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoInferenceApp()
    window.show()
    sys.exit(app.exec_())