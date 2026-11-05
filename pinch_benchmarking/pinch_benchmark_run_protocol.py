"""Interactive batch recorder for the protocol/experiment_plan.csv conditions.

The camera is opened independently for each trial, resetting the MediaPipe
tracker and pinch state machine. Every condition is saved as its own clip.
No labelling or evaluation is performed during capture.

Example: python pinch_benchmark_run_protocol.py --session session01 --camera 1
Preview the exact commands first with: --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from pinch_benchmark_common import manifest_by_id


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_.-")[:40]


def build_commands(args, plan):
    capture = Path(__file__).resolve().parent/"pinch_benchmark_capture.py"
    manifest = manifest_by_id(Path(args.session).expanduser().resolve())
    valid = set(args.experiments)
    for row in plan:
        experiment = row["experiment"].strip()
        if experiment not in valid:
            continue
        for replicate in range(1,int(row.get("replicates") or 1)+1):
            clip = f"{slug(experiment)}_{slug(row['condition'])}_{replicate:02d}"
            if clip in manifest:
                print(f"SKIP existing clip: {clip}")
                continue
            command = [sys.executable,str(capture),"--session",str(args.session),
                       "--clip-id",clip,"--experiment",experiment,
                       "--condition",row["condition"],
                       "--camera",str(args.camera),"--fps",str(args.fps),
                       "--duration",str(float(row["target_duration_s"])),
                       "--expected-cycles",str(int(row.get("expected_cycles") or 0)),
                       "--notes",row.get("hand_pose_or_action", "")]
            if row.get("approx_distance_cm", ""):
                command.extend(["--distance-cm",row["approx_distance_cm"]])
            if row.get("approx_orientation_deg", ""):
                command.extend(["--orientation-deg",row["approx_orientation_deg"]])
            for opt, val in (("--participant",args.participant),("--hand",args.hand),
                             ("--lighting",args.lighting),("--backend",args.backend)):
                if val:
                    command.extend([opt,val])
            command.append("--mirror" if args.mirror else "--no-mirror")
            if args.no_preview:
                command.append("--no-preview")
            if args.video_codec:
                command.extend(["--video-codec",args.video_codec])
            yield clip, row, command


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--session",required=True)
    p.add_argument("--plan",type=Path,default=Path(__file__).resolve().parent/"protocol/experiment_plan.csv")
    p.add_argument("--experiments",nargs="+",default=["pilot","distance","orientation","negative"],
                   help="Which experiment groups to record, e.g. distance orientation negative")
    p.add_argument("--camera",type=int,default=1)
    p.add_argument("--fps",type=float,default=30)
    p.add_argument("--participant",default="P01")
    p.add_argument("--hand",choices=["right","left","unknown"],default="right")
    p.add_argument("--lighting",default="normal")
    p.add_argument("--backend",choices=["auto","dshow","msmf"],default="")
    p.add_argument("--video-codec",choices=["mp4v","MJPG","FFV1"],default="mp4v")
    mirror=p.add_mutually_exclusive_group()
    mirror.add_argument("--mirror",dest="mirror",action="store_true",default=True,
                        help="Mirror horizontally like a selfie camera (default)")
    mirror.add_argument("--no-mirror",dest="mirror",action="store_false",
                        help="Keep the camera's unmirrored orientation")
    p.add_argument("--no-preview",action="store_true")
    p.add_argument("--dry-run",action="store_true",help="Print all commands without touching a camera")
    args=p.parse_args()
    if not args.plan.is_file():
        p.error(f"No protocol plan at {args.plan}")
    with args.plan.open(newline="",encoding="utf-8-sig") as source:
        protocol=list(csv.DictReader(source))
    commands=list(build_commands(args, protocol))
    if not commands:
        print("No unrecorded clips match the selected experiment groups.")
        return
    print(f"{len(commands)} protocol recordings planned; each clip is an independent trial")
    for i,(clip,row,cmd) in enumerate(commands,1):
        print(f"\n[{i}/{len(commands)}] {clip}: {row.get('hand_pose_or_action','')}")
        print("Command:",subprocess.list2cmdline(cmd))
        if args.dry_run:
            continue
        try:
            answer=input("Set up camera/hand, then ENTER to capture (s=skip, q=quit): ").strip().lower()
        except (EOFError,KeyboardInterrupt):
            print("\nStopped without starting next recording")
            break
        if answer == "q":
            break
        if answer == "s":
            continue
        code=subprocess.run(cmd,check=False).returncode
        if code:
            print(f"CAPTURE FAILED ({code}) for {clip}. Inspect setup before continuing")
            answer=input("Continue to next condition despite failure? [y/N]: ").strip().lower()
            if answer != "y":
                break
    print("\nCapture step finished. Annotate clips independently before evaluation.")


if __name__=="__main__":
    main()
