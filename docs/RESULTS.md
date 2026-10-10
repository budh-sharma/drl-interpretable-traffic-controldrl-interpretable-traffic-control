# Reference execution results

This is the executed preparation CPU run. The standalone notebook was also executed from a clean folder. Students still need their own rerun and contribution evidence.

Full training: 600 episodes. Frozen evaluation: 300 rollouts. Learning and evaluation are separate.

| Demand | Controller | Queue delay / arrival (s) | Reduction vs Actuated (%) |
| --- | --- | ---: | ---: |
| balanced | Fixed | 13.920 | -19.09 |
| balanced | Actuated | 11.688 | 0.00 |
| balanced | QueueGreedy | 10.629 | 9.06 |
| balanced | DQN | 10.149 | 13.17 |
| balanced | DRQ | 286.671 | -2352.59 |
| balanced | DRSQ | 9.878 | 15.49 |
| balanced | DRHQ | 9.828 | 15.92 |
| tidal | Fixed | 49.703 | -364.68 |
| tidal | Actuated | 10.696 | 0.00 |
| tidal | QueueGreedy | 9.327 | 12.80 |
| tidal | DQN | 8.903 | 16.76 |
| tidal | DRQ | 380.372 | -3456.17 |
| tidal | DRSQ | 8.851 | 17.25 |
| tidal | DRHQ | 8.758 | 18.12 |

The metric counts only waiting accumulated within the simulation horizon. Read throughput and pending vehicles in evaluation.csv alongside it. Positive reduction means lower delay. DRQ has severe residual queues and is retained as a failed variant.

## Two inspectable decisions

- balanced, t=300 s: NS score 9.034749, EW score 3.780013. Chosen group: NS.
- tidal, t=900 s: NS score 1.349616, EW score 12.572784. Chosen group: EW.

## Evidence integrity

Thirteen unit tests pass. All 300 rollouts conserve vehicles. Both decisions reconstruct their scores from contributions. Model checkpoint loading and finite parameters pass validation. Three DRHQ models pass 7,200 derivative-sign checks with zero violations. Sixteen diagnostic rollouts cover empty traffic, bursts, reversal and overload, conserve vehicles and leave the model parameters unchanged. Reference manifest hashes identify the inputs to every displayed summary.

Paper values, local measurements and personal contribution evidence are distinct. The local percentages do not reproduce the paper’s 19.4% result.
