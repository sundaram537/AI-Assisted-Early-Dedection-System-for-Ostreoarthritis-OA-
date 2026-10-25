"""
MediaPipe gait analysis prototype
---------------------------------
Measures both knee angles, observed knee-flexion ranges, trunk inclination,
a simple pelvic-height asymmetry proxy, heuristic step events, cadence,
step timing, and calibrated step-length/speed estimates.

IMPORTANT:
- This is a research/engineering prototype, NOT a clinical measurement tool.
- Use a fixed side-view camera, full body and both feet visible, level floor.
- Set PIXELS_PER_METER using a known distance measured in the SAME plane as
  the person's feet. Without calibration, metric step lengths/speed/distance
  are intentionally reported as unavailable (NaN), not invented.
- The foot-contact detector is heuristic and must be checked against video.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ------------------------- USER SETTINGS -------------------------
VIDEO_PATH = "C:\gaitposture\5319441-uhd_2160_3840_25fps.mp4"
OUTPUT_CSV = r"C:\gaitposture\gait_analysis_corrected.csv"

MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
VISIBILITY_THRESHOLD = 0.65
EMA_ALPHA = 0.30

# IMPORTANT: measure a known horizontal distance in the video at foot level.
# Example: if a 2.0 m marked segment spans 420 pixels, set 420 / 2.0 = 210.
# Leave as None until you calibrate. Never assume MediaPipe image coordinates
# are metres.
PIXELS_PER_METER = None

# Contact-event detector settings. Tune after visually checking foot tracks.
MIN_STEP_INTERVAL_SEC = 0.30
CONTACT_PROMINENCE_NORM = 0.004
SHOW_VIDEO = True
# ----------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def xyz(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)

def angle_3d(a, b, c):
    """Internal angle at b, in degrees; a-b-c straight is approximately 180°."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-9:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def angle_2d(a, b, c):
    """2D angle for display/checking; coordinates may be pixels or normalized."""
    return angle_3d([a[0], a[1], 0], [b[0], b[1], 0], [c[0], c[1], 0])

def ema(values, alpha=0.3):
    """EMA that restarts after missing values rather than bridging gaps."""
    out = []
    state = None
    for val in values:
        if val is None or not np.isfinite(val):
            out.append(np.nan)
            state = None
        else:
            state = float(val) if state is None else alpha * float(val) + (1-alpha) * state
            out.append(state)
    return out

def detect_contact_peaks(y_values, fps, min_interval_sec=0.30,
                         min_prominence=0.004):
    """
    Heuristic foot-ground contact candidates from local maxima of image y.
    Image y increases downwards, so the foot near the ground tends to have
    larger y. This is not a validated heel-strike detector.
    """
    y = np.asarray(y_values, dtype=float)
    if len(y) < 5:
        return []

    # Smooth small landmark jitter with a short moving average.
    window = max(3, int(round(0.10 * fps)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    filled = y.copy()
    valid = np.isfinite(filled)
    if valid.sum() < 5:
        return []
    # Interpolate gaps only for event detection; original rows remain unchanged.
    indices = np.arange(len(filled))
    filled[~valid] = np.interp(indices[~valid], indices[valid], filled[valid])
    smooth = np.convolve(filled, kernel, mode="same")

    min_distance = max(1, int(round(min_interval_sec * fps)))
    candidates = []
    radius = max(3, int(round(0.25 * fps)))

    for i in range(1, len(smooth)-1):
        if smooth[i] >= smooth[i-1] and smooth[i] > smooth[i+1]:
            left = max(0, i-radius)
            right = min(len(smooth), i+radius+1)
            local_min = min(np.min(smooth[left:i+1]), np.min(smooth[i:right]))
            prominence = smooth[i] - local_min
            if prominence >= min_prominence:
                candidates.append((i, prominence))

    # Keep the stronger peak when candidates are too close together.
    candidates.sort(key=lambda item: item[1], reverse=True)
    chosen = []
    for idx, _ in candidates:
        if all(abs(idx - old) >= min_distance for old in chosen):
            chosen.append(idx)
    return sorted(chosen)

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Edit VIDEO_PATH at the top of this script to your actual video path."
    )

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps <= 0:
    fps = 30.0
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Video: {VIDEO_PATH}")
print(f"FPS: {fps:.2f} | Frames: {frame_total}")
print("Processing full-resolution frames for pose estimation...")

records = []
frame_no = 0
previous_flex = {"left": None, "right": None}

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
) as pose:

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        time_s = (frame_no - 1) / fps
        h, w = frame.shape[:2]

        # Detect on original frame; resize only the display copy.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)

        row = {
            "Frame": frame_no, "Time_sec": time_s,
            "Left_Knee_Internal_deg": np.nan,
            "Right_Knee_Internal_deg": np.nan,
            "Left_Knee_Flexion_deg": np.nan,
            "Right_Knee_Flexion_deg": np.nan,
            "Trunk_Inclination_deg": np.nan,
            "Pelvic_Hip_Height_Difference_pct": np.nan,
            "Left_Heel_X_norm": np.nan, "Left_Heel_Y_norm": np.nan,
            "Right_Heel_X_norm": np.nan, "Right_Heel_Y_norm": np.nan,
            "Left_Visibility_Min": np.nan, "Right_Visibility_Min": np.nan,
            "Frame_Quality": "No pose"
        }

        if result.pose_landmarks and result.pose_world_landmarks:
            img_lm = result.pose_landmarks.landmark
            world_lm = result.pose_world_landmarks.landmark

            def get_img(name):
                lm = img_lm[int(getattr(mp_pose.PoseLandmark, name))]
                return np.array([lm.x, lm.y], dtype=float), float(lm.visibility)

            def get_world(name):
                return xyz(world_lm[int(getattr(mp_pose.PoseLandmark, name))])

            # Required visibility checks for each knee.
            left_names = ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]
            right_names = ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
            left_vis = [get_img(n)[1] for n in left_names]
            right_vis = [get_img(n)[1] for n in right_names]
            row["Left_Visibility_Min"] = min(left_vis)
            row["Right_Visibility_Min"] = min(right_vis)

            left_ok = min(left_vis) >= VISIBILITY_THRESHOLD
            right_ok = min(right_vis) >= VISIBILITY_THRESHOLD

            if left_ok:
                li = angle_3d(get_world("LEFT_HIP"), get_world("LEFT_KNEE"),
                              get_world("LEFT_ANKLE"))
                if np.isfinite(li):
                    row["Left_Knee_Internal_deg"] = li
                    row["Left_Knee_Flexion_deg"] = 180.0 - li

            if right_ok:
                ri = angle_3d(get_world("RIGHT_HIP"), get_world("RIGHT_KNEE"),
                              get_world("RIGHT_ANKLE"))
                if np.isfinite(ri):
                    row["Right_Knee_Internal_deg"] = ri
                    row["Right_Knee_Flexion_deg"] = 180.0 - ri

            # Trunk inclination: shoulder-midpoint to hip-midpoint against vertical.
            shoulder_mid = (get_world("LEFT_SHOULDER") + get_world("RIGHT_SHOULDER")) / 2
            hip_mid = (get_world("LEFT_HIP") + get_world("RIGHT_HIP")) / 2
            torso = shoulder_mid - hip_mid
            torso_len = np.linalg.norm(torso)
            if torso_len > 1e-8:
                # World y is vertical in MediaPipe coordinates; vertical reference.
                vertical_component = abs(float(torso[1])) / torso_len
                vertical_component = np.clip(vertical_component, 0.0, 1.0)
                row["Trunk_Inclination_deg"] = float(np.degrees(np.arccos(vertical_component)))

            # Pelvic proxy: left-right hip height difference normalized by
            # estimated mean thigh+shank length. More useful in a front view;
            # NOT a validated clinical pelvic asymmetry measurement.
            lhip, rhip = get_world("LEFT_HIP"), get_world("RIGHT_HIP")
            lkn, rkn = get_world("LEFT_KNEE"), get_world("RIGHT_KNEE")
            lank, rank = get_world("LEFT_ANKLE"), get_world("RIGHT_ANKLE")
            leg_lengths = [
                np.linalg.norm(lhip-lkn) + np.linalg.norm(lkn-lank),
                np.linalg.norm(rhip-rkn) + np.linalg.norm(rkn-rank)
            ]
            mean_leg = np.mean(leg_lengths)
            if mean_leg > 1e-8:
                # Absolute vertical difference as a percentage of leg length.
                row["Pelvic_Hip_Height_Difference_pct"] = (
                    abs(lhip[1] - rhip[1]) / mean_leg * 100.0
                )

            # Heel positions from normalized image landmarks, for step-event detection.
            for side in ("LEFT", "RIGHT"):
                heel, _ = get_img(f"{side}_HEEL")
                row[f"{side.title()}_Heel_X_norm"] = float(heel[0])
                row[f"{side.title()}_Heel_Y_norm"] = float(heel[1])

            row["Frame_Quality"] = (
                "Both knees visible" if left_ok and right_ok
                else "One/both knees low visibility"
            )

            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Display filtered-ish instantaneous angles (not the CSV EMA).
            y_text = 32
            for side, label in (("Left", "L"), ("Right", "R")):
                val = row[f"{side}_Knee_Flexion_deg"]
                if np.isfinite(val):
                    cv2.putText(frame, f"{label} flexion: {val:.1f} deg",
                                (20, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2, cv2.LINE_AA)
                    y_text += 32

        records.append(row)

        if SHOW_VIDEO:
            show = frame
            if w > 1280:
                scale = 1280 / w
                show = cv2.resize(frame, (1280, int(h * scale)))
            cv2.imshow("MediaPipe Gait Analysis - press q to stop", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user.")
                break

cap.release()
cv2.destroyAllWindows()

if not records:
    raise RuntimeError("No video frames were processed.")

df = pd.DataFrame(records)

# Smooth angle signals independently; gaps remain gaps.
df["Left_Knee_Flexion_Smoothed_deg"] = ema(
    df["Left_Knee_Flexion_deg"].tolist(), EMA_ALPHA
)
df["Right_Knee_Flexion_Smoothed_deg"] = ema(
    df["Right_Knee_Flexion_deg"].tolist(), EMA_ALPHA
)

# Observed ranges are descriptive only, not clinical ROM.
left_valid = df["Left_Knee_Flexion_Smoothed_deg"].dropna()
right_valid = df["Right_Knee_Flexion_Smoothed_deg"].dropna()
left_rom = float(left_valid.max() - left_valid.min()) if len(left_valid) else np.nan
right_rom = float(right_valid.max() - right_valid.min()) if len(right_valid) else np.nan

# Trunk and pelvic summary: median is less sensitive to single-frame spikes.
trunk_summary = df["Trunk_Inclination_deg"].dropna()
pelvic_summary = df["Pelvic_Hip_Height_Difference_pct"].dropna()
trunk_median = float(trunk_summary.median()) if len(trunk_summary) else np.nan
pelvic_median = float(pelvic_summary.median()) if len(pelvic_summary) else np.nan

# Heuristic contact events from local maxima of heel y, where image y increases down.
left_events = detect_contact_peaks(
    df["Left_Heel_Y_norm"].to_numpy(), fps,
    MIN_STEP_INTERVAL_SEC, CONTACT_PROMINENCE_NORM
)
right_events = detect_contact_peaks(
    df["Right_Heel_Y_norm"].to_numpy(), fps,
    MIN_STEP_INTERVAL_SEC, CONTACT_PROMINENCE_NORM
)

events = []
for i in left_events:
    events.append({"frame_idx": i, "time": i / fps, "side": "Left",
                   "x_norm": df.iloc[i]["Left_Heel_X_norm"]})
for i in right_events:
    events.append({"frame_idx": i, "time": i / fps, "side": "Right",
                   "x_norm": df.iloc[i]["Right_Heel_X_norm"]})
events.sort(key=lambda e: e["time"])

# Remove near-duplicate left/right events that often occur from occlusion/noise.
clean_events = []
for event in events:
    if not clean_events or event["time"] - clean_events[-1]["time"] >= MIN_STEP_INTERVAL_SEC:
        clean_events.append(event)
    else:
        # Keep the candidate with more plausible/available x data; if both,
        # keep the one with the more extreme y contact frame.
        prev = clean_events[-1]
        if np.isfinite(event["x_norm"]) and not np.isfinite(prev["x_norm"]):
            clean_events[-1] = event

step_times = np.diff([e["time"] for e in clean_events])
step_time_mean = float(np.median(step_times)) if len(step_times) else np.nan
duration = max((len(df) - 1) / fps, 0.0)
total_steps = len(clean_events)
cadence = (total_steps / duration * 60.0) if duration > 0 else np.nan

# Metric spatial measures are intentionally unavailable until calibration is supplied.
step_lengths_m = []
if PIXELS_PER_METER is not None and PIXELS_PER_METER > 0:
    # Convert normalized image x difference to pixels using frame width.
    # Successive contacts are used; camera must be perpendicular to walking direction.
    for prev, curr in zip(clean_events[:-1], clean_events[1:]):
        if np.isfinite(prev["x_norm"]) and np.isfinite(curr["x_norm"]):
            dx_pixels = abs(curr["x_norm"] - prev["x_norm"]) * float(video_width)
            step_lengths_m.append(dx_pixels / PIXELS_PER_METER)

if step_lengths_m:
    average_step_length = float(np.mean(step_lengths_m))
    step_length_difference = (
        abs(float(np.mean(step_lengths_m[::2])) - float(np.mean(step_lengths_m[1::2])))
        if len(step_lengths_m) >= 2 else np.nan
    )
    # This is an alternating-step proxy, not a clinically validated left/right estimate.
    left_lengths = [step_lengths_m[i] for i, e in enumerate(clean_events[1:]) if e["side"] == "Left" and i < len(step_lengths_m)]
    right_lengths = [step_lengths_m[i] for i, e in enumerate(clean_events[1:]) if e["side"] == "Right" and i < len(step_lengths_m)]
    left_step_length = float(np.mean(left_lengths)) if left_lengths else np.nan
    right_step_length = float(np.mean(right_lengths)) if right_lengths else np.nan
    distance_m = float(np.sum(step_lengths_m))
    gait_speed = distance_m / duration if duration > 0 else np.nan
    asym_denom = (left_step_length + right_step_length) / 2
    step_asymmetry_pct = (
        abs(left_step_length - right_step_length) / asym_denom * 100
        if np.isfinite(left_step_length) and np.isfinite(right_step_length) and asym_denom > 1e-9
        else np.nan
    )
else:
    average_step_length = step_length_difference = np.nan
    left_step_length = right_step_length = np.nan
    distance_m = gait_speed = step_asymmetry_pct = np.nan

# Add per-frame contact flags to CSV.
df["Left_Contact_Candidate"] = False
df["Right_Contact_Candidate"] = False
for i in left_events:
    if i < len(df):
        df.loc[i, "Left_Contact_Candidate"] = True
for i in right_events:
    if i < len(df):
        df.loc[i, "Right_Contact_Candidate"] = True

# Write detailed per-frame data and summary file.
out_dir = os.path.dirname(OUTPUT_CSV)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

summary = {
    "Left Knee Angle (median internal)": float(df["Left_Knee_Internal_deg"].median()),
    "Right Knee Angle (median internal)": float(df["Right_Knee_Internal_deg"].median()),
    "Left Knee Observed Flexion Range (deg)": left_rom,
    "Right Knee Observed Flexion Range (deg)": right_rom,
    "Trunk Inclination (median deg)": trunk_median,
    "Pelvic/Hip Height Difference Proxy (median %)": pelvic_median,
    "Step Time (median sec)": step_time_mean,
    "Left Step Length (m)": left_step_length,
    "Right Step Length (m)": right_step_length,
    "Average Step Length (m)": average_step_length,
    "Step Length Difference (m)": step_length_difference,
    "Left-Right Step Asymmetry Proxy (%)": step_asymmetry_pct,
    "Gait Speed (m/s)": gait_speed,
    "Cadence (steps/min)": cadence,
    "Total Steps (heuristic contacts)": total_steps,
    "Estimated Distance (m)": distance_m,
    "Video Duration (sec)": duration,
    "Pixels per Meter Calibration": PIXELS_PER_METER,
    "Left Contact Candidates": len(left_events),
    "Right Contact Candidates": len(right_events),
}

summary_path = os.path.splitext(OUTPUT_CSV)[0] + "_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("GAIT ANALYSIS SUMMARY (PROTOTYPE; NOT CLINICALLY VALIDATED)\n")
    f.write("=" * 66 + "\n")
    for key, value in summary.items():
        if isinstance(value, (float, np.floating)):
            f.write(f"{key}: {value:.3f}\n" if np.isfinite(value) else f"{key}: NOT AVAILABLE\n")
        else:
            f.write(f"{key}: {value}\n")
    f.write("\nInterpretation warnings:\n")
    f.write("- Foot contacts are heuristic candidates and must be checked against the video.\n")
    f.write("- Metric step lengths, speed, and distance require valid pixel-to-meter calibration.\n")
    f.write("- Pelvic height difference is a proxy, not a validated clinical asymmetry metric.\n")
    f.write("- Knee range is observed range during this clip, not full clinical ROM.\n")

print("\n" + "=" * 66)
print("FINAL GAIT ANALYSIS (PROTOTYPE)")
print("=" * 66)
for key, value in summary.items():
    if isinstance(value, (float, np.floating)):
        display = f"{value:.3f}" if np.isfinite(value) else "NOT AVAILABLE"
    else:
        display = str(value)
    print(f"{key:<48}: {display}")
print("=" * 66)
print(f"Per-frame CSV: {OUTPUT_CSV}")
print(f"Summary file: {summary_path}")
print("\nImportant: inspect contact flags overlaid on the video and calibrate before using metres.")
