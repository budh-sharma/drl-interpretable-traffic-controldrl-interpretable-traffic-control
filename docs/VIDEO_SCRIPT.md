# Group 87 — seven-minute presentation script

Target rehearsal length: **6:30**. The script has about 761 words, leaving time to change slides and point at evidence. The hard limit is seven minutes. Record a rehearsal and adjust the pacing before recording the final video.

Budhdi covers slides 1–11. Sumeet covers 12–20 and 22. Both speak on slide 21. This assigns presentation sections, not historical project contributions. Fill the contribution statements only after doing and verifying the work.

Before recording, insert the personal screenshots and real links. The performance narration uses the tested local reference values. If your Colab run differs, update the chart/table values and the spoken numbers from `results/ppt_values.json`. Do not read editing instructions or placeholder text aloud.

| Slide | Time window | Speaker | Topic |
|---|---|---|---|
| 1 | 0:00–0:10 | Budhdi | Learning an Interpretable Traffic Signal Control Policy |
| 2 | 0:10–0:27 | Budhdi | Why This Problem Matters |
| 3 | 0:27–0:44 | Budhdi | Traffic Signal Control as an MDP |
| 4 | 0:44–1:01 | Budhdi | Phases, Conflicts & Clearance Intervals |
| 5 | 1:01–1:17 | Budhdi | What Makes the Policy Regulatable |
| 6 | 1:17–1:39 | Budhdi | Precedence Score and Monotonicity |
| 7 | 1:39–1:57 | Budhdi | A Worked Precedence Decision |
| 8 | 1:57–2:16 | Budhdi | How Deep RL Trains an Interpretable Controller |
| 9 | 2:16–2:37 | Budhdi | One Training Update |
| 10 | 2:37–2:55 | Budhdi | DRQ vs DRSQ vs DRHQ |
| 11 | 2:55–3:09 | Budhdi | CMA-ES vs PPO vs Q-Learning |
| 12 | 3:09–3:24 | Sumeet | Experimental Setup |
| 13 | 3:24–3:45 | Sumeet | Results: Performance vs Interpretability |
| 14 | 3:45–4:07 | Sumeet | Our Local Experiment |
| 15 | 4:07–4:22 | Sumeet | Limitations and Future Work |
| 16 | 4:22–4:38 | Sumeet | Colab Run and Correctness Proof |
| 17 | 4:38–4:56 | Sumeet | Measured Learning Curves |
| 18 | 4:56–5:18 | Sumeet | Example 1: Balanced Demand |
| 19 | 5:18–5:40 | Sumeet | Example 2: Demand Reversal |
| 20 | 5:40–6:02 | Sumeet | Decision and Monotonicity Proof |
| 21 | 6:02–6:18 | Both | Individual Proof of Work |
| 22 | 6:18–6:30 | Sumeet | Conclusion and References |

## Slide 1 — Learning an Interpretable Traffic Signal Control Policy

We are Group 87. This presentation reviews the paper and then shows an executed, inspectable implementation of its learning approach.

## Slide 2 — Why This Problem Matters

The aim is to reduce traffic delay while keeping signal changes understandable. Traffic engineers need to see why an action wins and how to adjust it. A black-box neural actuator makes that difficult.

## Slide 3 — Traffic Signal Control as an MDP

The state describes traffic and current signals. An action selects a valid phase combination. The environment advances traffic and enforces clearance. Reward reflects delay. Learning seeks higher discounted return within those movement constraints.

## Slide 4 — Phases, Conflicts & Clearance Intervals

The environment enforces yellow and all-red clearance. The flags also let G account for switching cost. A learned score cannot bypass the transition rules.

## Slide 5 — What Makes the Policy Regulatable

Each action has a precedence score. The largest score wins. At a fixed checkpoint, each input has a consistent direction of influence. This supports inspection without proving every outcome safe or optimal.

## Slide 6 — Precedence Score and Monotonicity

Equation 1 is the paper function. Our adaptation puts signed weights outside powers. Input normalization is fixed; positive clearance factors are learned. At a saved checkpoint the derivative sign follows the weight.

## Slide 7 — A Worked Precedence Decision

Figure 2 gives 22.53 for westbound through and left terms combined, versus 19.31 eastbound. Westbound wins. The earlier wording incorrectly called 22.53 the left-turn score alone. These are paper values, separate from our measured examples.

## Slide 8 — How Deep RL Trains an Interpretable Controller

G acts from the start, apart from exploration. A DQN teacher learns off-policy from those transitions and supplies targets for G. This separates the learner from the actuator and keeps the deployed decision inspectable during training.

## Slide 9 — One Training Update

Our reward is minus added queue waiting divided by 100. Forty vehicle-seconds gives minus 0.40. With gamma 0.95 and next value minus 2, the target is minus 2.30, or minus 0.40 at termination. The scaling affects learning, while displayed delay remains seconds per arrival. DRHQ teaches G the winner from Q.

## Slide 10 — DRQ vs DRSQ vs DRHQ

DRQ fits numerical action values. DRSQ fits soft preferences. DRHQ fits only the best action. Matching the winner asks less of the restricted score function. We compare all three and retain the unsuccessful outcomes.

## Slide 11 — CMA-ES vs PPO vs Q-Learning

CMA-ES can tune a strong policy but needs many full episodes. PPO converges more slowly. These are paper comparisons. Our code executes DQN and its three regulatable variants.

## Slide 12 — Experimental Setup

The paper uses observed Utah demand in SUMO, eleven valid phase pairs and three demand profiles. Its actuated baseline matters. Changing the simulator or baseline changes what an improvement percentage means.

## Slide 13 — Results: Performance vs Interpretability

The paper reports up to 19.4 percent lower delay for DRHQ against actuated control. DQN performs better but uses a neural actuator. Figure 4 covers thirty seeded trials. Low and medium demand improve after one episode. High demand takes about twenty.

## Slide 14 — Our Local Experiment

This is an algorithm demonstration, not a SUMO reproduction. Four FIFO queues share two through groups. Three seeds produce 600 training episodes. Freeze the models and evaluate identical held-out arrivals in 300 rollouts. Waiting inside the horizon includes unfinished vehicles, but not their future waiting.

## Slide 15 — Limitations and Future Work

The paper assumes reliable sensing and tests one simulated intersection. The authors propose warm-starting from an existing controller. Our demonstration also omits turning, spillback and pedestrians. Field deployment needs wider validation.

## Slide 16 — Colab Run and Correctness Proof

The notebook provides execution evidence. Thirteen tests cover clearance, accounting, gradients and checkpoints. Sixteen diagnostics cover empty traffic, bursts, reversal and overload. Overload leaves queues as expected. Evaluation checks confirm unchanged model parameters.

## Slide 17 — Measured Learning Curves

Training curves average three seeds and come from saved CSVs. Exploration changes during training, so final claims use frozen evaluation. Poor DRQ performance remains in the full output. The screenshot identifies the actual run.

## Slide 18 — Example 1: Balanced Demand

The balanced reference means are 11.688 seconds for Actuated and 9.828 for DRHQ. Subtract and divide by Actuated to get 15.92 percent. Pending vehicles and throughput are displayed too. The interval covers held-out demand variation conditional on three trained models. Cite the balanced rows in summary.csv.

## Slide 19 — Example 2: Demand Reversal

Demand reverses at 600 seconds. The displayed delay, pending queues and throughput summarize frozen evaluation. DRHQ has 18.12 percent lower delay in this reference run. Its interval measures demand variation conditional on the trained models. DRQ fails badly and remains in the notebook.

## Slide 20 — Decision and Monotonicity Proof

Show the implemented Section 11 decision cell. The balanced North/South contributions sum to about 9.035, above East/West at 3.780, so North/South wins. The tidal example selects East/West. Full-precision terms reconstruct scores. The derivative checks concern fixed checkpoints and clearance categories, not road safety.

## Slide 21 — Individual Proof of Work

We each show our own completed task and explain its check. Budhdi: insert your actual task and finding. Sumeet: insert your actual task and finding. The screenshots and contribution log identify the work.

## Slide 22 — Conclusion and References

DRHQ lets a strong learner guide an inspectable actuator. The references, submitted code and actual recording link support the analysis and its evidence.

## Replace these words before recording

- Slide 21: each person gives their actual task, file/cell and finding. Keep both statements brief and show their own evidence.
- Slides 18–20: confirm that the Colab measurements match the displayed reference values. If they differ, replace the values rather than claim a match.
- Slide 22: insert the actual code/notebook and video links, with working access.

## Keep the video readable

Use the actual slideshow and completed notebook. Point at one equation or result at a time. Record at a readable source resolution, export under 10 MB, and play the exported file at normal size. Check the equations, table values and screenshot text, particularly after compression. A source recording that looks clear does not prove that the final exported video is readable.
