from __future__ import annotations
from pathlib import Path
import os, json
import av, numpy as np, torch
ODE_OPT={"method":"midpoint","options":{"step_size":2/32}}

def atomic_save_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)

def decode_audio_tensor(path: Path, sample_rate: int) -> torch.Tensor:
    """Decode the MP4 audio with PyAV, avoiding TorchCodec system-FFmpeg ABI issues."""
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"The input MP4 has no audio stream: {path}")
        resampler = None
        for frame in container.decode(audio=0):
            if resampler is None:
                resampler = av.AudioResampler(
                    format="fltp",
                    layout=frame.layout.name,
                    rate=sample_rate,
                )
            for output_frame in resampler.resample(frame):
                chunks.append(output_frame.to_ndarray())
        if resampler is not None:
            for output_frame in resampler.resample(None):
                chunks.append(output_frame.to_ndarray())
    if not chunks:
        raise RuntimeError(f"No audio samples were decoded from: {path}")
    waveform = np.concatenate(chunks, axis=1)
    return torch.from_numpy(np.ascontiguousarray(waveform)).float()

def decode_video_tensor(path: Path) -> torch.Tensor:
    """Decode every video frame as uint8 NCHW RGB, matching the processor input."""
    frame_list: list[torch.Tensor] = []
    frame_tensor: torch.Tensor | None = None
    frame_count = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"The input MP4 has no video stream: {path}")
        declared_frames = int(container.streams.video[0].frames or 0)
        for frame in container.decode(video=0):
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            chw = torch.from_numpy(rgb).permute(2, 0, 1)
            if declared_frames > 0:
                if frame_tensor is None:
                    frame_tensor = torch.empty(
                        (declared_frames, *chw.shape), dtype=chw.dtype
                    )
                if frame_count >= declared_frames:
                    raise RuntimeError(
                        "The MP4 contains more decoded frames than its container metadata."
                    )
                if tuple(chw.shape) != tuple(frame_tensor.shape[1:]):
                    raise RuntimeError("The MP4 changes frame resolution mid-stream.")
                frame_tensor[frame_count].copy_(chw)
            else:
                frame_list.append(chw)
            frame_count += 1
    if frame_count == 0:
        raise RuntimeError(f"No video frames were decoded from: {path}")
    if frame_tensor is not None:
        return frame_tensor[:frame_count]
    return torch.stack(frame_list, dim=0)
