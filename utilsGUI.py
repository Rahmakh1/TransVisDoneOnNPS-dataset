import cv2
import numpy as np
import torch
from utils.augmentations import letterbox_temporal
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device


def load_model(weights_path=r"C:\Users\Lenovo\NPS\best.pt", device=None):
   
    # Select device (GPU if available, otherwise CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = attempt_load(weights_path, map_location=device)
    model.eval()  # Set model to evaluation mode

    # Use half-precision (FP16) if supported by the hardware
    half = device.type != 'cpu'  # Half precision only supported on CUDA
    model.half() 

    return model, device



def run_inference(video_path, model, device, img_size=640, num_frames=5, skip_frames=0, stride=32,
                  conf_thres=0.001, iou_thres=0.6, augment=False, single_cls=False):
    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # Prepare variables
    frames = []
    all_predictions = []
    resize_info = []  # To store resize and padding info for each frame
    clip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        # Check if we have enough frames to form a clip
        if len(frames) == num_frames:
            # Preprocess the frames
            preprocessed_frames = []
            for frame in frames:
                # Resize and letterbox
                resized_frame, ratio, (dw, dh) = letterbox_temporal([frame], new_shape=img_size, stride=stride)
                resized_frame = resized_frame[0]  # Unpack single frame
                resize_info.append((ratio, (dw, dh)))  # Store scaling ratio and padding

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
            half = device.type != 'cpu'  # Half precision only supported on CUDA
            clip = clip.to(device, non_blocking=True)
            clip = clip.half() if half else clip.float()

            # Run model
            with torch.no_grad():
                out, _ = model(clip, augment=augment)

            # Apply Non-Maximum Suppression
            predictions = non_max_suppression(out, conf_thres, iou_thres, classes=0, agnostic=single_cls)

            # Collect predictions for this clip
            all_predictions.append(predictions)
            clip_count += 1

            # Remove only 1 frame from the buffer to create overlapping clips
            frames = frames[1:]

    print("count clips", clip_count)

    # Release the video capture
    cap.release()

    return all_predictions, resize_info  # Return predictions and resize info


def save_predictions(predictions, output_file="predictions.txt"):
    with open(output_file, "w") as f:
        for i, Frame_preds in enumerate(predictions):
            f.write(f"Frame {i}:\n")
            if Frame_preds is not None:
                for det in Frame_preds:  # det is a tensor of shape [N, 6]
                    if det.numel() > 0:  # Check if det is not empty
                        det = det.cpu().numpy()  # Move tensor to CPU and convert to numpy
                        for box in det:  # Iterate over each bounding box
                            x1, y1, x2, y2, conf, cls = box  # Unpack the 6 values
                            f.write(f"Class: {int(cls)}, Confidence: {conf:.4f}, BBox: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]\n")
                    else:
                        f.write("No detections in this Frame.\n")
            else:
                f.write("No detections.\n")


def main(video_path, weights_path, output_file=None, img_size=640, num_frames=5, skip_frames=0, 
         conf_thres=0.001, iou_thres=0.6, augment=False, single_cls=False):
  
    # Load the model
    print("Loading model...")
    model, device = load_model(weights_path)

    # Run inference on the video
    print("Running inference...")
    predictions = run_inference(video_path, model, device, img_size=img_size, num_frames=num_frames,
                               skip_frames=skip_frames,  conf_thres=conf_thres,
                               iou_thres=iou_thres, augment=augment, single_cls=single_cls)

    # Save  predictions
    if output_file:
        print(f"Saving predictions to {output_file}...")
        save_predictions(predictions, output_file)

def apply_gamma_correction(frame, gamma):
    """Apply gamma correction to a frame."""
    
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def apply_histogram_equalization(frame):
    """Apply histogram equalization to a frame."""
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)
    # Convert back to BGR (to maintain consistency with other processing steps)
    return cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)



if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\Lenovo\NPS\NPS_dataset\Videos\test\Clip_43.mov"
    weights_path = r"C:\Users\Lenovo\NPS\best.pt"
    output_file = r"C:\Users\Lenovo\NPS\runs\predictions.txt"

    main(video_path, weights_path, output_file=output_file)