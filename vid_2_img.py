import cv2
import numpy as np
import os
import random

def is_sharp(image):
    """Returns sharpness value of the image based on Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_threshold(video_path, sample_frames=200, quantile=0.3):
    """
    Estimate a blur threshold from a subset of frames.
    quantile = 0.3 means we keep frames sharper than the 30 percent weakest.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 20.0  # fallback

    indices = list(range(total_frames))
    if len(indices) > sample_frames:
        indices = random.sample(indices, sample_frames)

    sharpness_values = []

    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        sharpness_values.append(is_sharp(frame))

    cap.release()

    if not sharpness_values:
        return 20.0

    sharpness_values = np.array(sharpness_values, dtype=np.float64)
    threshold = float(np.quantile(sharpness_values, quantile))
    return threshold

def extract_sharp_frames(
    video_path,
    output_folder,
    threshold=None,
    sharp_frame_interval=20,
    flip_horizontal=False,
    flip_vertical=False,
    resize_short_side=None,
):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    sharp_frame_count = 0

    max_variance = 0.0
    sharpest_frame = None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Auto threshold if not provided
    if threshold is None:
        print("Estimating sharpness threshold...")
        threshold = estimate_threshold(video_path)
        print(f"Using automatic sharpness threshold: {threshold:.2f}")
    else:
        print(f"Using fixed sharpness threshold: {threshold:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Optional downscale to speed up processing
        if resize_short_side is not None:
            h, w = frame.shape[:2]
            if h < w:
                new_h = resize_short_side
                new_w = int(w * (new_h / h))
            else:
                new_w = resize_short_side
                new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        variance = is_sharp(frame)

        if variance > threshold:
            sharp_frame_count += 1

            if variance > max_variance:
                max_variance = variance
                sharpest_frame = frame.copy()

            if sharp_frame_count % sharp_frame_interval == 0 and sharpest_frame is not None:
                out_frame = sharpest_frame

                # Conditional flipping
                if flip_horizontal:
                    out_frame = cv2.flip(out_frame, 1)
                if flip_vertical:
                    out_frame = cv2.flip(out_frame, 0)

                save_path = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(save_path, out_frame)
                saved_count += 1

                max_variance = 0.0
                sharpest_frame = None

        frame_count += 1

    cap.release()
    print(f"Total video frames: {frame_count}")
    print(f"Saved sharp frames: {saved_count}")

if __name__ == "__main__":
    video_path = "v.mp4"
    output_folder = "./output_frames"

    extract_sharp_frames(
        video_path=video_path,
        output_folder=output_folder,
        threshold=None,               # None means automatic estimation
        sharp_frame_interval=20,
        flip_horizontal=False,        # I would keep both False for COLMAP
        flip_vertical=False,
        resize_short_side=720,        # Downscale for faster sharpness checks
    )
