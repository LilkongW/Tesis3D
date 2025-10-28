import cv2
import numpy as np
import os
import sys

# --- 📂 CONFIGURATION ---
# WARNING: Videos in this folder will be PERMANENTLY overwritten.
# Please make a backup of this folder before running the script.
VIDEOS_FOLDER = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/Majo/ROI_videos_640x480"

# Set the intensity of the whitening effect (0.0 to 1.0).
WHITEN_INTENSITY = 0.2
# -------------------------

# Global list to store the points for the current video's ROI
roi_points = []
frame_for_selection = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to capture points for the ROI."""
    global roi_points, frame_for_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"  Point added: ({x}, {y}). Total points: {len(roi_points)}")
        
        cv2.circle(frame_for_selection, (x, y), 5, (0, 255, 0), -1)
        if len(roi_points) > 1:
            cv2.line(frame_for_selection, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
        
        cv2.imshow("Define ROI - Press ENTER to confirm", frame_for_selection)

def select_roi_for_video(first_frame, video_name):
    """Opens a window for the user to select a polygonal ROI on a frame."""
    global roi_points, frame_for_selection
    
    roi_points = [] # Reset points for each new video
    frame_for_selection = first_frame.copy()
    
    window_name = "Define ROI - Press ENTER to confirm"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n======================================================")
    print(f"ACTION REQUIRED for video: {video_name}")
    print("  - LEFT-CLICK to define the area to keep.")
    print("  - Press ENTER to confirm the area and start processing.")
    print("  - Press 'r' to reset points for this video.")
    print("  - Press 'q' to quit the entire script.")
    print("======================================================")
    
    while True:
        cv2.imshow(window_name, frame_for_selection)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): # Quit
            cv2.destroyAllWindows()
            return None
        
        if key == ord('r'): # Reset
            roi_points = []
            frame_for_selection = first_frame.copy()
            print("  🔄 Points reset. Please define the area again.")
        
        if key == 13: # Enter key
            if len(roi_points) < 3:
                print("  ⚠️ Please select at least 3 points to form an area.")
            else:
                cv2.destroyAllWindows()
                print("  ✅ ROI confirmed.")
                return roi_points

def apply_whitening_effect(original_image, points, intensity):
    """Applies the whitening effect outside the selected polygon."""
    height, width = original_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    white_overlay = np.ones_like(original_image, dtype=np.uint8) * 255
    
    inverted_mask = cv2.bitwise_not(mask)
    
    blended_overlay = cv2.addWeighted(original_image, 1 - intensity, white_overlay, intensity, 0)
    
    result = original_image.copy()
    np.copyto(result, blended_overlay, where=inverted_mask[:, :, None].astype(bool))

    return result

def batch_process_and_overwrite_videos(target_dir):
    """Processes all videos in a directory, applying an effect, and overwrites them."""
    if not os.path.isdir(target_dir):
        print(f"❌ Error: The directory does not exist: '{target_dir}'")
        return

    # --- SAFETY WARNING ---
    print("====================================================================")
    print("🛑 WARNING: THIS SCRIPT WILL PERMANENTLY OVERWRITE ORIGINAL VIDEOS. 🛑")
    print(f"   Target Directory: {target_dir}")
    print("   This action cannot be undone.")
    print("   Please ensure you have a backup before proceeding.")
    print("====================================================================")
    
    try:
        input("Press Enter to begin, or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n🚫 Process cancelled by user.")
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print(f"❌ No video files found in '{target_dir}'")
        return

    print(f"\n✅ Found {len(video_files)} videos to process.")
    
    for video_filename in video_files:
        original_path = os.path.join(target_dir, video_filename)
        cap = cv2.VideoCapture(original_path)
        
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_filename}. Skipping.")
            continue
        
        ret, first_frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read first frame of {video_filename}. Skipping.")
            cap.release()
            continue
            
        points = select_roi_for_video(first_frame, video_filename)
        
        if points is None:
            print("\n🚫 Quitting batch process as requested.")
            cap.release()
            break

        print(f"Processing '{video_filename}'...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        name, ext = os.path.splitext(video_filename)
        temp_output_path = os.path.join(target_dir, f"{name}_temp{ext}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            processed_frame = apply_whitening_effect(frame, points, WHITEN_INTENSITY)
            out.write(processed_frame)
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 50)
            sys.stdout.write(f"\r  [{'#' * progress}{'.' * (50 - progress)}] {frame_count}/{total_frames}")
            sys.stdout.flush()

        cap.release()
        out.release()
        
        # --- CRITICAL STEP: REPLACE ORIGINAL FILE ---
        try:
            os.remove(original_path)
            os.rename(temp_output_path, original_path)
            print(f"\n✔️ Finished. Video '{original_path}' has been overwritten.")
        except Exception as e:
            print(f"\n❌ ERROR replacing file '{original_path}': {e}")
            print(f"   The temporary file was saved as '{temp_output_path}'")

    print("\n\n🎉 All videos processed!")

if __name__ == "__main__":
    batch_process_and_overwrite_videos(VIDEOS_FOLDER)
    cv2.destroyAllWindows()