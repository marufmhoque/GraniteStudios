"""Multi-format video processing & editing engine.

FFmpeg-backed (bundled via imageio-ffmpeg, so no system install is required)
with OpenCV powering the computer-vision-driven operations (auto-crop,
stabilization). Usable directly as a Python API — see pipeline.VideoPipeline
and pipeline.JobManager — or through the optional local REST API in api.py.
"""
