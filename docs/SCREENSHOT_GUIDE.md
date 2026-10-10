# Group 87 — screenshots and final submission

The revised deck has **22 slides**. Slides 1–15 explain and analyze the paper, including the score, a worked example and the training update. Slides 16–22 show the implementation, measurements, personal evidence and references.

Ten original slide visuals remain unchanged: original slides 1, 2, 4, 5, 6, 8, 9, 10, 11 and 14. The worked-example error is corrected in a new slide. The unverified redrawn results curves are replaced with the paper's verifiable findings. Local charts read measured CSVs rather than typed performance claims.

## Run and capture

Open `DRL_Group87_Colab.ipynb` in Google Colab and select a CPU runtime. Run all cells with **FULL_RUN = True**. The saved outputs show a clean local preparation run, not a live Colab run or personal student proof. Keep your own completed notebook and the evidence ZIP from Section 16.

| Slide | Your real capture | What it establishes |
|---|---|---|
| 16 | Section 8 test summary and Section 13 proof summary, with notebook name | 13 tests, full run counts, conservation and frozen diagnostic checks |
| 17 | Section 9 completed training output or Section 10 learning curves | Which algorithms and seeds actually ran |
| 18 | Section 10 balanced summary rows, including failed DRQ | Your measured comparison, units and paired interval |
| 19 | Section 12 tidal trace and Section 10 tidal summary rows | Demand reversal response and measured comparison |
| 20 | Sections 11/12 contribution table and Section 13 monotonicity result | Actual action scores, reconstruction and derivative checks |
| 21 | Each person's own completed run or a meaningful verified check | Actual individual work, with a file/cell and explainable finding |
| 22 | Editable link fields, no screenshot required | Real shared code/notebook and Google Drive video links |

Use two captures inside a proof space when necessary. Crop tightly around useful output. Delete the green screenshot label and gray instructions after inserting your image. Keep its proportions. Do not compress a large notebook page into an unreadable tiny image. Expand or split outputs instead.

The charts and tables are editable. Compare your actual values with `results/ppt_values.json`. For a chart, use PowerPoint's Edit Data command to replace the embedded values if necessary. Keep the stated source/run identity accurate. Figure 2 arithmetic and the small TD/loss example are illustrations, separate from the measured traffic results.

## Personal contribution record

In Section 13, fill `CONTRIBUTOR_NAME` and `ACTUAL_CONTRIBUTION` with the work you actually completed, then rerun that cell. It creates `results/contributor_run.json` with execution details and result hashes. If the fields are empty, it correctly says that personal details are pending. Each person should keep their own evidence and complete `docs/CONTRIBUTIONS.md`. A name in the record alone does not establish a contribution: the person must perform and explain the work.

Both people should discuss and verify the paper and implementation. The proposed division is only a working plan. Budhdi can verify the environment, teacher updates and balanced example; Sumeet can verify frozen evaluation, uncertainty and the tidal example. Use your actual division if different.

## Rubric coverage

| Requirement | Slides / evidence |
|---|---|
| Objective and motivation, 2 marks | Slides 1–3 |
| Techniques and algorithmic approaches, 2 marks | Slides 4–11, notebook Sections 3–5 and 14 |
| Results, interpretation, limitations, future work and conclusion, 3 marks | Slides 12–15, 18–20 and 22 |
| Clear recording and precise proof for each contributor, 5 marks | Slides 16–22, personal screenshots, actual contribution log, shared evidence and timed narration |

The rubric does not require a full SUMO reproduction. The local project demonstrates the learning approach in a smaller simulator. It does not establish the paper's 19.4% reduction again. The paper's 30% CMA-ES and 19.4% DRHQ findings and the local 15.92%/18.12% reference measurements belong to different experiments.

## Checks before the single fresh submission

- Correct Group 87, both full names and BITS IDs on the first slide.
- Proof spaces on slides 16–21 filled with readable personal Colab captures, including both contributors on slide 21. Remove remaining instructional text after adding pictures.
- Real code/notebook link and actual video Drive link on the last slide. Verify access while signed out.
- Record both contributors, clearly explain algorithm and evidence, and stay at or below seven minutes. The included script targets 6:30.
- Both PPT and video at or below 10 MB. The empty-proof PPT is small, but inserted screenshots will increase its size. Recheck file sizes and final video readability after compression.
- Use the required filename `DRL_AssignmentName_GroupNo`. This deck uses `DRL_LearningInterpretableTrafficSignalControlPolicy_87.pptx`; use the corresponding stem for the video.
- Fresh portal submission by the professor's stated October 11, 2026 deadline. The instructions allow only one submission and do not accept email or Teams submissions.

A complete deck supports a strong submission. The remaining student evidence, link access and recording quality still affect the grade, and the assessor decides the marks.
