import os
from multiprocessing import cpu_count
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm
import argparse


def process_single_case(
    case_id,
    label_dir,
    video_dir,
    out_frame_dir,
    out_label_dir,
    target_fps=1,
    resize=(256, 256),
):
    case_folder = f"case_{case_id:03d}"
    label_file = os.path.join(label_dir, case_folder, "tasks.csv")
    video_folder = os.path.join(video_dir, case_folder)

    if not os.path.exists(label_file):
        print(f"Label file {label_file} does not exist")
        return

    # Load CSV
    df = pd.read_csv(label_file)

    # Create output directories if they don't exist
    frame_out_path = os.path.join(out_frame_dir, str(case_id))
    annot_out_path = os.path.join(out_label_dir, f"{case_id:03d}-phase.txt")
    os.makedirs(frame_out_path, exist_ok=True)
    os.makedirs(os.path.dirname(annot_out_path), exist_ok=True)

    # Initialize frame counter
    export_frame_counter = 0
    annot_records = []

    # Get list of video parts sorted by filename
    part_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    for i, part_file in enumerate(part_files, start=1):
        video_path = os.path.join(video_folder, part_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = int(fps / target_fps)  # Calculate frame interval based on fps

        for frame_counter in tqdm(
            range(0, frame_count, interval),
            desc=f"Case {case_id}",
            position=remaining_cases.index(case_id) + 1,
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            ret, frame = cap.read()
            if not ret:
                break

            if resize:
                frame_shape = frame.shape
                h, w = frame_shape[:2]
                crop_w = int(h * 3.75 / 3)
                crop_h = int(0.92 * h)
                frame = frame[:crop_h, (w - crop_w) // 2 : (w + crop_w) // 2]
                frame = cv2.resize(frame, resize)

            # Save the frame as a JPEG file
            frame_filename = f"{export_frame_counter}.jpg"
            cv2.imwrite(os.path.join(frame_out_path, frame_filename), frame)

            # Get frame label
            frame_time = frame_counter / fps
            frame_part = i

            # Determine the phase for the current frame
            criteria_1 = (
                (df["start_time"] <= frame_time)
                & (df["stop_time"] >= frame_time)
                & (df["start_part"] == frame_part)
                & (df["stop_part"] == frame_part)
            )
            criteria_2 = (
                (df["start_time"] <= frame_time)
                & (df["start_part"] == frame_part)
                & (df["stop_part"] > frame_part)
            )
            criteria_3 = (
                (df["stop_time"] >= frame_time)
                & (df["start_part"] < frame_part)
                & (df["stop_part"] == frame_part)
            )
            possible_phases = df[criteria_1 | criteria_2 | criteria_3][
                "groundtruth_taskname"
            ].values
            if len(possible_phases) == 1:
                phase = possible_phases[0]
            elif len(possible_phases) > 1:
                if all(phase == possible_phases[0] for phase in possible_phases):
                    phase = possible_phases[0]
                else:
                    raise ValueError(
                        f"Multiple phases found for frame {frame_counter} in case {case_id} in part {frame_part}: {possible_phases}"
                    )
            else:
                phase = "Other"

            annot_records.append({"Frame": export_frame_counter, "Phase": phase})
            export_frame_counter += 1

        cap.release()

    annot_df = pd.DataFrame(annot_records)
    annot_df.to_csv(annot_out_path, sep="\t", index=False)


def process_all_cases(
    label_directory,
    video_directory,
    frame_output_directory,
    label_output_directory,
    remaining_cases,
):
    nprocess = cpu_count()
    number_of_cases = len(remaining_cases)
    print(f"Processing {number_of_cases} cases, with {nprocess} CPU cores")
    with tqdm(total=number_of_cases, desc="Total Progress", unit="case") as pbar:
        with Pool(nprocess) as pool:
            results = [
                pool.apply_async(
                    process_single_case,
                    (
                        case_id,
                        label_directory,
                        video_directory,
                        frame_output_directory,
                        label_output_directory,
                    ),
                )
                for case_id in remaining_cases
            ]
            for result in results:
                result.wait()
                if not result.successful():
                    print(result.get())
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_directory",
        type=str,
        required=True,
        help="Directory containing the labels",
        default="/datasets/surgvu24/labels",
    )
    parser.add_argument(
        "--video_directory",
        type=str,
        required=True,
        help="Directory containing the videos",
        default="/datasets/surgvu24/surgvu24",
    )
    parser.add_argument(
        "--frame_output_directory",
        type=str,
        required=True,
        help="Directory to save the frames",
        default="/datasets/surgvu24/processed/frames",
    )
    parser.add_argument(
        "--label_output_directory",
        type=str,
        required=True,
        help="Directory to save the labels",
        default="/datasets/surgvu24/processed/annot",
    )
    parser.add_argument(
        "--remaining_cases",
        type=int,
        nargs="+",
        help="List of case IDs to process",
        default=list(range(155)),
    )
    args = parser.parse_args()
    # Usage
    label_directory = args.label_directory
    video_directory = args.video_directory
    frame_output_directory = args.frame_output_directory
    label_output_directory = args.label_output_directory
    remaining_cases = args.remaining_cases

    process_all_cases(
        label_directory,
        video_directory,
        frame_output_directory,
        label_output_directory,
        remaining_cases,
    )
