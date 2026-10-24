# Next

1. Finish and evaluate E18's longer training run. E17's qualified recovery results are recorded. Record measured quality, training time and contention in `EXPERIMENTS.md`; preserve the optimizer-reset qualification for E17. Read `RECOVERY.md` before restarting either experiment.
2. Run E19's larger training budget after allocating suitable compute. Use full-state checkpoints, session-independent orchestration and external snapshots. The config is prepared; the run has not started.
3. Port the released Sudoku checkpoints to MLX and score them on the same boards with this evaluator. This separates the method's performance from this implementation's training.
4. Compile both samplers and rerun synchronized benchmarks to measure how much of the small-batch gap is Python dispatch.
5. Evaluate a real-text corpus, such as a TinyStories subset, against an autoregressive baseline at equal model size and training budget.

Width-512 and model-sampled corruption were already tested (E10/E12 and E9). Their outcomes are in `EXPERIMENTS.md`.
