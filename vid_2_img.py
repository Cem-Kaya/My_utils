import cv2
import numpy as np
import os

def resize_for_processing(frame, max_side=None):
    """Resize keeping aspect ratio so the longest side is max_side."""
    if max_side is None:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def is_sharp(image):
    """Sharpness value of the image based on Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_video_for_sharpness(video_path, resize_max_side=None, quantile=0.3):
    """
    First pass.
    Compute sharpness for every frame, find threshold by quantile,
    and count how many frames are above that threshold.
    """
    cap = cv2.VideoCapture(video_path)
    variances = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_proc = resize_for_processing(frame, resize_max_side)
        var = is_sharp(frame_proc)
        variances.append(var)
        total_frames += 1

    cap.release()

    if not variances:
        return 20.0, 0, total_frames

    var_arr = np.array(variances, dtype=np.float64)
    threshold = float(np.quantile(var_arr, quantile))
    sharp_mask = var_arr > threshold
    sharp_count = int(sharp_mask.sum())

    return threshold, sharp_count, total_frames

def extract_sharp_frames(
    video_path,
    output_folder,
    threshold=None,
    sharp_frame_interval=20,
    target_saved_frames=None,
    flip_horizontal=False,
    flip_vertical=False,
    resize_max_side=1080,
    save_max_side=1080,
    quantile=0.3,
):
    # First pass: threshold and sharp frame stats
    print("Analyzing video for sharpness stats...")
    if threshold is None:
        threshold, sharp_count, total_frames = analyze_video_for_sharpness(
            video_path,
            resize_max_side=resize_max_side,
            quantile=quantile,
        )
    else:
        # If threshold is fixed, still need stats to choose interval if target_saved_frames used
        threshold_tmp, sharp_count, total_frames = analyze_video_for_sharpness(
            video_path,
            resize_max_side=resize_max_side,
            quantile=quantile,
        )
        total_frames = total_frames  # just to be explicit

    # Decide interval
    if target_saved_frames is not None and sharp_count > 0:
        interval = max(1, sharp_count // target_saved_frames)
    else:
        interval = max(1, sharp_frame_interval)

    print(f"Total video frames: {total_frames}")
    print(f"Sharpness threshold: {threshold:.2f}")
    print(f"Frames above threshold: {sharp_count}")
    print(f"Using save interval: every {interval} sharp frames")

    # Make output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Second pass: actually save frames
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    sharp_seen = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_proc = resize_for_processing(frame, resize_max_side)
        var = is_sharp(frame_proc)

        if var > threshold:
            sharp_seen += 1

            if sharp_seen % interval == 0:
                out_frame = frame_proc

                if flip_horizontal:
                    out_frame = cv2.flip(out_frame, 1)
                if flip_vertical:
                    out_frame = cv2.flip(out_frame, 0)

                out_frame = resize_for_processing(out_frame, save_max_side)

                save_path = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(save_path, out_frame)
                saved_count += 1

        frame_index += 1

    cap.release()

    print(f"Saved sharp frames: {saved_count}")

if __name__ == "__main__":
    video_path = "v.mp4"
    output_folder = "./output_frames"

    extract_sharp_frames(
        video_path=video_path,
        output_folder=output_folder,
        threshold=None,            # auto from quantile
        sharp_frame_interval=20,   # fallback if target_saved_frames is None
        target_saved_frames=800,   # what you asked for
        flip_horizontal=False,
        flip_vertical=False,
        resize_max_side=1080,      # process around 1080p
        save_max_side=1080,        # save around 1080p
        quantile=0.3,              # drop the worst 30 percent blur
    )
