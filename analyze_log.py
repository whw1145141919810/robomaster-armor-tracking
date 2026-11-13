"""Compare paired raw and filtered measurements from a tracking CSV."""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


AXES = ("x", "y", "z", "distance")
LABELS = ("X", "Y", "Z", "距离")
RAW_COLUMNS = tuple(f"raw_{axis}_m" for axis in AXES)
FILTERED_COLUMNS = tuple(f"filtered_{axis}_m" for axis in AXES)
REQUIRED_COLUMNS = {"frame", "time_s", "track_id", "status", *RAW_COLUMNS, *FILTERED_COLUMNS}


@dataclass
class Measurement:
    frame: int
    time_s: float
    raw: tuple[float, ...]
    filtered: tuple[float, ...]


def read_measurements(
    path: Path, start_s: float, end_s: float
) -> tuple[dict[str, list[Measurement]], Counter]:
    tracks: dict[str, list[Measurement]] = {}
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV 缺少列 {', '.join(sorted(missing))}")

        for row in reader:
            try:
                time_s = float(row["time_s"])
                if not math.isfinite(time_s) or time_s < 0:
                    raise ValueError("time_s 必须是有限的非负数")
                if not start_s <= time_s <= end_s:
                    continue

                status = row["status"]
                if status not in ("measured", "coasting", "lost"):
                    raise ValueError(f"未知的跟踪状态 {status!r}")
                counts[status] += 1
                if status != "measured":
                    continue

                track_id = row["track_id"]
                if not track_id:
                    raise ValueError("measured 行缺少 track_id")
                raw = tuple(float(row[column]) for column in RAW_COLUMNS)
                filtered = tuple(float(row[column]) for column in FILTERED_COLUMNS)
                if not all(map(math.isfinite, (*raw, *filtered))):
                    raise ValueError("位置和距离必须是有限值")

                measurement = Measurement(int(row["frame"]), time_s, raw, filtered)
                samples = tracks.setdefault(track_id, [])
                if samples and (
                    measurement.frame <= samples[-1].frame or time_s < samples[-1].time_s
                ):
                    raise ValueError("同一跟踪 ID 的帧号须递增，时间须按顺序记录")
                samples.append(measurement)
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV 第 {reader.line_num} 行无效。{error}") from error
    return tracks, counts


def report_static(samples: list[Measurement]) -> None:
    print("分量 | 原始均值 (m) | 滤波均值 (m) | 原始样本标准差 (mm) | 滤波样本标准差 (mm) | 标准差降幅")
    for index, label in enumerate(LABELS):
        raw = [sample.raw[index] for sample in samples]
        filtered = [sample.filtered[index] for sample in samples]
        means = f"{label} | {statistics.mean(raw):.4f} | {statistics.mean(filtered):.4f}"
        if len(samples) < 2:
            print(f"{means} | — | — | —")
            continue

        raw_std = statistics.stdev(raw) * 1000
        filtered_std = statistics.stdev(filtered) * 1000
        reduction = f"{(1 - filtered_std / raw_std):+.1%}" if raw_std else "—"
        print(f"{means} | {raw_std:.3f} | {filtered_std:.3f} | {reduction}")


def report_moving(samples: list[Measurement]) -> None:
    print("分量 | 原始范围 (m) | 滤波范围 (m)")
    for index, label in enumerate(LABELS):
        raw = [sample.raw[index] for sample in samples]
        filtered = [sample.filtered[index] for sample in samples]
        print(f"{label} | {min(raw):.4f} 至 {max(raw):.4f} | {min(filtered):.4f} 至 {max(filtered):.4f}")

    distances = [math.dist(sample.raw[:3], sample.filtered[:3]) for sample in samples]
    rms_difference = math.sqrt(statistics.mean(distance**2 for distance in distances))
    print(f"同帧原始与滤波三维位置差的均方根为 {rms_difference * 1000:.3f} mm。")

    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        raw_displacement = math.dist(first.raw[:3], last.raw[:3])
        filtered_displacement = math.dist(first.filtered[:3], last.filtered[:3])
        print(f"首末实测帧的三维位移为原始 {raw_displacement:.4f} m，滤波 {filtered_displacement:.4f} m。")
        consecutive_pairs = [
            (previous, current)
            for previous, current in zip(samples, samples[1:])
            if current.frame == previous.frame + 1
        ]
        if consecutive_pairs:
            raw_path = sum(math.dist(a.raw[:3], b.raw[:3]) for a, b in consecutive_pairs)
            filtered_path = sum(math.dist(a.filtered[:3], b.filtered[:3]) for a, b in consecutive_pairs)
            print(f"{len(consecutive_pairs)} 对相邻实测帧的累计路径长为原始 {raw_path:.4f} m，滤波 {filtered_path:.4f} m。")


def main() -> None:
    parser = argparse.ArgumentParser(description="比较跟踪日志中同一帧的 PnP 和滤波位置。")
    parser.add_argument("csv", type=Path, help="跟踪程序输出的 CSV")
    parser.add_argument("--mode", choices=("static", "moving"), required=True, help="目标静止或移动")
    parser.add_argument("--start-s", type=float, default=0.0, help="选段起始秒数，包含该时刻")
    parser.add_argument("--end-s", type=float, help="选段结束秒数，包含该时刻")
    args = parser.parse_args()
    if not math.isfinite(args.start_s) or args.start_s < 0:
        parser.error("--start-s 必须是有限的非负数")
    if args.end_s is not None and (not math.isfinite(args.end_s) or args.end_s < args.start_s):
        parser.error("--end-s 必须是有限数，且大于或等于 --start-s")

    end_s = args.end_s if args.end_s is not None else math.inf
    try:
        tracks, counts = read_measurements(args.csv, args.start_s, end_s)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    if not tracks:
        parser.error("选段内没有同时包含原始和滤波位置的实测帧")

    print(f"选段内实测 {counts['measured']} 帧，短时外推 {counts['coasting']} 帧，丢失 {counts['lost']} 帧。")
    report = report_static if args.mode == "static" else report_moving
    for track_id, samples in tracks.items():
        print(f"\n跟踪 ID {track_id}，{len(samples)} 个配对样本，{samples[0].time_s:.3f} 至 {samples[-1].time_s:.3f} s")
        report(samples)


if __name__ == "__main__":
    main()
