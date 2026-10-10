# Algorithm walkthrough

## 1. The control problem

Four incoming queues represent North, South, East and West through traffic. The controller chooses one of two nonconflicting groups: North/South or East/West. Each action occupies ten seconds. A change uses two seconds of yellow and one second of all-red first, leaving seven seconds of green. Staying green provides ten seconds of service. Arrivals continue during clearance.

Vehicles join a FIFO queue. Each active direction can discharge one vehicle per second. We count every vehicle that remains in a queue after service as one vehicle-second of waiting. This deliberately simple simulator makes each result traceable. It does not claim physical traffic realism.

The observation contains six features for each direction, followed by four clearance flags for each candidate action. The features are queued count, one-second scheduled approaching count, cumulative wait of queued vehicles, mean wait of queued vehicles, queue per lane, and synthetic approach speed. A scheduled approaching arrival receives 12 m/s and otherwise speed is zero. A single lane per direction makes queue per lane identical to queue count. These redundancies mirror the paper's feature vocabulary while showing a limitation of the simplified setting. The observation compresses the full FIFO state, so it is not a claim of exact Markov sufficiency.

The approach detector exposes the next scheduled one-second arrivals to every learned controller. It is an idealized sensor, not a live data source. Fixed and Actuated intentionally use their usual restricted inputs.

## 2. DQN teacher

The teacher maps 32 observations to two Q-values using two hidden layers of 32 leaky-ReLU units. Replay separates the training minibatch from the most recent interaction. The target network changes only every 100 updates.

For each sampled transition:

```
y = reward + gamma * (1 - terminal) * max(target_Q(next_state))
```

Reward is minus the queue-delay increment divided by 100. The division is a constant scale for optimization and never changes the displayed metrics. The teacher minimizes Huber loss with Adam. Terminal transitions have no bootstrap term.

DQN deploys its teacher directly. DRQ, DRSQ and DRHQ deploy G from the first interaction, apart from epsilon exploration, and train a separate teacher off-policy using those interactions. They are not post-training distillation of a fully trained DQN. Each variant has its own replay, teacher and actor.

## 3. Interpretable actuator

For an action a, the local project uses:

```
G_a(s) = c_a(flag) * sum(w[a,lane,feature] * x[lane,feature] ** p[a,lane,feature])
p = 1 + sigmoid(power_raw)
c_a(flag) = sum(flag[j] * (0.1 + 0.9 * sigmoid(clear_raw[a,j])))
```

Only lanes served by the candidate action contribute. Input scales are fixed constants, not statistics fitted on evaluation data. Inputs remain nonnegative and are not clipped, so the score remains differentiable on its valid domain. Signed coefficients allow increasing or decreasing influence. At a fixed checkpoint:

```
dG_a/dx_i = c_a * w_i * p_i * x_i ** (p_i - 1)
```

The sign is determined by w_i because c_a is positive and 1 < p_i < 2. At x_i=0 the derivative is zero. The direction can change between training checkpoints if a coefficient changes sign, but within a deployed checkpoint it remains consistent across inputs. Changing a clearance category is discrete, so monotonicity here concerns continuous traffic variables conditional on the flag. The contribution table sums exactly to the selected score.

There are 24 coefficients, 24 exponent parameters and 8 clearance factors: 56 trainable actor parameters. One-hot clearance exponents are fixed at 1. We support the full/partial/permissive/none flag vocabulary, but this two-action environment activates only full and none. Inactive clearance parameters receive zero gradient.

The project's family uses `w*x**p`, whereas Equation 1 in the paper writes `(w*x)**p`. This avoids nonreal powers of negative weighted inputs and permits direct inspection of signed effects. It is a disclosed restriction/adaptation, not an exact reproduction of the original function or its 352-parameter intersection.

## 4. Three ways to teach G

| Variant | Target | Loss | Why compare it |
| --- | --- | --- | --- |
| DRQ | TD target y for the replay action | Squared error `(G(s,a)-y)**2` | Attempts action-value fitting with a restricted function |
| DRSQ | `softmax(Q(s,:))` | Cross-entropy against `softmax(G(s,:))` | Fits relative preference strength |
| DRHQ | One-hot vector for `argmax Q(s,:)` | Cross-entropy against `softmax(G(s,:))` | Fits the preferred action |

Teacher targets are constants for the actor update. Teacher and actor optimizers do not share parameters. Policy training uses two independently sampled minibatches per teacher update. `np.argmax` resolves an exact tie toward action 0 for learned models. QueueGreedy stays on its current action on a tie.

## 5. Metrics and fairness

The headline local metric is total waiting accumulated within the 1,200-second horizon divided by all vehicles that arrived in that horizon. It includes waiting already incurred by unfinished vehicles but excludes their future waiting after the horizon. It must not be called full average trip delay. Completed-only mean and p95 wait, arrived/departed/pending counts, throughput fraction and switching count accompany it.

All controllers receive identical pre-generated arrivals for a given demand seed. Training uses seeds `10000 + 1000 * training_seed + episode`, while evaluation uses 90000-90009. Evaluation is greedy, frozen and never updates weights. Three independent training seeds are 11, 22 and 33. Baselines need no training and their evaluation rows appear once, not once per learning seed.

For each algorithm and scenario, average the three learned models within each held-out demand stream. Bootstrap the ten paired demand streams 2,000 times and recompute the ratio of mean delays against Actuated. The resulting interval expresses demand-stream variation conditional on these trained models. It does not fully quantify training uncertainty. `training_seed_summary.csv` separately shows variation between training seeds.

No model or demonstration seed is selected by evaluation performance. Demonstrations always use the first declared training seed, 11, and traffic seed 90000.

## 6. What this implementation establishes

It demonstrates the learning loop, its gradients, the difference between three actor targets, frozen evaluation, conservation of vehicles, clearance mechanics and score explanations. It does not establish road deployment safety, superiority across cities, or replication of the paper's 19.4% result. PPO and CMA-ES belong to the paper review but are outside this project's executed algorithms.

## References

- Ault, Hanna and Sharon. *Learning an Interpretable Traffic Signal Control Policy*. AAMAS 2020. https://arxiv.org/abs/1912.11023
- Authors' original experiment repository: https://github.com/jault/StateStreetSumo
- Mnih et al. *Human-level control through deep reinforcement learning*. Nature 2015. https://doi.org/10.1038/nature14236
