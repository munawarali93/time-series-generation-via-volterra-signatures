"""Animate 24 actual trajectories saved by create_flow_video.py."""
from pathlib import Path
import json

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "videos"
source = OUT / "flow_matching_trajectories.npz"
if not source.exists():
    raise FileNotFoundError("Run create_flow_video.py first to save the model trajectories.")
with np.load(source) as saved:
    all_paths = saved["path"]
path = all_paths[:, :24].copy()
assert path.ndim == 3 and path.shape[2] == 2 and np.isfinite(path).all()
steps = len(path) - 1
centers = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
groups = ((path[-1, :, None] - centers[None]) ** 2).sum(-1).argmin(-1)
palette = np.array(["#2266B3", "#D6603C", "#008878", "#8459A5"])
colors = palette[groups]
navy = "#15345C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "text.color": navy, "axes.labelcolor": navy,
                     "xtick.color": navy, "ytick.color": navy})
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
fig.text(.055, .94, "Learned trajectories: Gaussian noise to GMM",
         fontsize=25, weight="bold")
fig.text(.055, .89, "Follow 24 individual samples as the trained velocity field moves them",
         fontsize=14, color="#526579")
ax = fig.add_axes([.06, .21, .53, .62])
limit = max(3.6, float(np.abs(path).max()) + .3)
ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
       xlabel="Coordinate 1", ylabel="Coordinate 2")
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["bottom", "left"]].set_color("#B5C3D0")
ax.grid(alpha=.12)
ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=90,
           linewidths=1.5, color="#9EAAB6", zorder=1)
lines = [ax.plot([], [], color=c, lw=1.7, alpha=.8, zorder=2)[0] for c in colors]
ax.scatter(path[0, :, 0], path[0, :, 1], s=45, facecolors="none",
           edgecolors=colors, linewidths=1.3, zorder=3)
dots = ax.scatter(path[0, :, 0], path[0, :, 1], s=33, c=colors,
                  edgecolors="white", linewidths=.45, zorder=4)

legend_ax = fig.add_axes([.64, .49, .34, .32])
legend_ax.axis("off")
legend_ax.text(0, 1, "Reading the paths", fontsize=19, weight="bold",
               transform=legend_ax.transAxes)
handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                  markeredgecolor=navy, markersize=8, label="Gaussian starting point"),
           Line2D([], [], marker="o", linestyle="none", color=navy,
                  markersize=7, label="Current position"),
           Line2D([], [], color=navy, lw=2, label="Path already traveled"),
           Line2D([], [], marker="x", linestyle="none", color="#9EAAB6",
                  markersize=9, label="Target cluster center")]
legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.03, .89),
                  frameon=False, labelspacing=1.25, fontsize=13)
fig.text(.64, .405, "At each step, the model uses\nposition and time to predict velocity.",
         fontsize=15, linespacing=1.6)
fig.text(.64, .30, "No target endpoint is supplied\nduring generation.",
         fontsize=14, linespacing=1.6)
fig.text(.64, .20, "Colors indicate the nearest final cluster.\n"
         "They are assigned after generation.",
         fontsize=12, color="#526579", linespacing=1.6)
clock = fig.text(.075, .08, "t = 0.00", fontsize=21, weight="bold", color="#2266B3")
fig.text(.25, .08, r"$X_{k+1}=X_k+\Delta t\,v_\theta(X_k,t_k)$", fontsize=19)
fig.text(.66, .08, f"24 trajectories  |  {steps} Euler steps", fontsize=12)

fps = 30
frame_indices = [0] * 30 + list(range(1, steps + 1)) + [steps] * 90
video = OUT / "learned_trajectories.mp4"
writer = imageio_ffmpeg.write_frames(str(video), (1280, 720), fps=fps,
    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
    output_params=["-movflags", "+faststart"], ffmpeg_log_level="error")
writer.send(None)
try:
    for frame_no, index in enumerate(frame_indices):
        for j, line in enumerate(lines):
            line.set_data(path[:index + 1, j, 0], path[:index + 1, j, 1])
        dots.set_offsets(path[index])
        clock.set_text(f"t = {index / steps:.2f}")
        fig.canvas.draw()
        writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        if frame_no % 120 == 0:
            print(f"Encoded {frame_no}/{len(frame_indices)} frames", flush=True)
finally:
    writer.close()
plt.close(fig)
(OUT / "trajectory_video_metadata.json").write_text(json.dumps({
    "source": source.name, "selected_sample_indices": list(range(24)),
    "frames": len(frame_indices), "fps": fps,
    "duration_seconds": len(frame_indices) / fps, "resolution": [1280, 720],
    "nearest_final_cluster_indices": groups.tolist(),
    "note": "Saved learned ODE trajectories, with growing trails. No retraining or endpoint interpolation."
}, indent=2) + "\n")
print(f"Saved {video}", flush=True)
