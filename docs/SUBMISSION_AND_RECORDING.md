# Resubmission and recording

The revised supplied instructions require the paper's motivation, techniques, results, interpretation, limitations, conclusion, and precise proof of work for every contributor. Group 87 is allocated the traffic-control paper. The first slide already includes the names and BITS IDs from the old presentation.

## What remains yours to do

1. Upload the standalone notebook to Colab. Choose a CPU runtime and run all cells. Keep the executed notebook and download the evidence ZIP.
2. Review the source, run outputs and explanations yourself. Record what each person actually reviewed, ran, changed and explained. Slide 21 contains fields for each contributor’s actual task and evidence. Replace it with completed, truthful evidence. Do not present the generated code or reference execution as work you personally performed before this project existed. Follow your course's rules on assisted work.
3. Check that your local results match the reference run closely. If you change configuration, update the measured results in the deck from your CSVs rather than keeping numbers from another configuration.
4. Record a fresh full-screen slideshow. Explain the paper and show the local code, its output, and one decision explanation. Both contributors should make their individual work identifiable. The deck's speaker notes and `VIDEO_SCRIPT.md` provide a timed outline.
5. Upload the actual video to Google Drive and set permissions so the evaluator can view it. Replace the last-slide video-link text with that real link. Add your real project repository link if you publish it. No link has been invented or uploaded on your behalf.
6. Submit both PPT and video under the required naming convention, with each file below 10 MB and the video no longer than seven minutes. The email says the deadline is October 11, 2026 and earlier submissions are drafts, so complete the fresh portal submission. The document allows one submission. Check before the final submit action.

## Recording without blur

Use Slide Show mode, not the slide editor or a PDF thumbnail view. Start at 1280x720 or 1920x1080 and keep the source recording at good quality. Hide toolbars. Speak while slides are static and avoid continuous pointer motion. Record a short 20-second sample first, watch it at 100% scale and confirm you can read the formulas and chart labels.

A seven-minute 10 MB file permits only about 190 kbps for everything combined. Do not blindly crush the full-resolution source to fit. The included deck has large type and sparse layouts to help low-bitrate recording. Rehearse the included 6:30 script, use mono speech audio, and preserve a high-quality original in Drive. Submit the file required by your portal as well as the link required on the last slide.

An example final encoding command, where FFmpeg is available, is:

```bash
ffmpeg -i source_recording.mp4 -vf "scale=1280:720,fps=10" -c:v libx264 -preset slow -b:v 160k -pass 1 -an -f mp4 NUL
ffmpeg -i source_recording.mp4 -vf "scale=1280:720,fps=10" -c:v libx264 -preset slow -b:v 160k -pass 2 -c:a aac -ac 1 -b:a 32k DRL_LearningInterpretableTrafficSignalControlPolicy_87.mp4
```

`NUL` is the Windows null destination. On Linux use `/dev/null`. This bitrate targets roughly 9.4 MB for 390 seconds, plus muxing overhead. It is a starting point, not a promise of readability. Measure actual size, inspect the encoded video, and shorten the recording if necessary. Upscaling the old 640x480 recording cannot restore missing letters.

## Evidence locations

- `results/tests.txt`: executed unit tests.
- `results/run.log`: training progress and frozen evaluation summary.
- `results/manifest.json`: actual settings, seeds, versions and SHA-256 checksums.
- `results/training.csv`: every episode's delay and losses.
- `results/evaluation.csv`: every frozen rollout, including pending vehicles.
- `results/summary.csv`: paired comparison and confidence intervals.
- `results/training_seed_summary.csv`: variation between independently trained models.
- `results/examples.json`: raw features, weights, powers and score terms for both examples.
- `results/trace_*.csv`: second-by-second queues and signal clearance.
- `results/models/*.npz`: all frozen checkpoints.
- `results/figures/`: plots and queue animation.

The supplied reference outputs come from a CPU execution in the preparation environment. Your recorded proof should show your own rerun, not claim that these were already your group's Colab outputs.

See `SCREENSHOT_GUIDE.md` for the exact slide-to-notebook capture mapping and `REVIEW_AND_QA.md` for questions to answer. Section 13 now saves diagnostics.csv, proof_summary.json, contributor_run.json and ppt_values.json.
