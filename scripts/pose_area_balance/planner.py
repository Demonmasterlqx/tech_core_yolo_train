from __future__ import annotations

from dataclasses import dataclass

from .metrics import AreaBin
from pose_dataset_build_utils import PoseDatasetSample


@dataclass(frozen=True)
class GenerationTask:
    source_sample: PoseDatasetSample
    target_bin: AreaBin
    slot_index: int
    output_stem: str


def generated_target_count_per_bin(raw_train_count: int, generated_multiplier: int, bin_count: int) -> int:
    if raw_train_count <= 0:
        raise ValueError("raw_train_count must be positive.")
    if generated_multiplier <= 0:
        raise ValueError("generated_multiplier must be positive.")
    if generated_multiplier != bin_count:
        raise ValueError(
            "This builder uses one generated sample per source per target bin, so "
            "runtime.generated_multiplier must equal metric.bin_count."
        )
    return raw_train_count


def build_generation_tasks(samples: list[PoseDatasetSample], bins: list[AreaBin], generated_multiplier: int) -> list[GenerationTask]:
    generated_target_count_per_bin(len(samples), generated_multiplier, len(bins))
    tasks: list[GenerationTask] = []
    for sample in samples:
        for area_bin in bins:
            output_stem = f"{sample.stem}_area_bin_{area_bin.index:02d}"
            tasks.append(
                GenerationTask(
                    source_sample=sample,
                    target_bin=area_bin,
                    slot_index=area_bin.index,
                    output_stem=output_stem,
                )
            )
    return tasks
