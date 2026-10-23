# Q-Nav Step 3 Audit: Quantum Accuracy Optimization Pipeline Analysis

## Executive Summary
This audit rigorously investigates the operational bottlenecks limiting the current Variational Quantum Circuit (VQC) validation accuracy to **26.20%** (Macro F1 = 0.0964, baseline random = 11.11%, majority baseline = 34.93%). Through mathematical inspection, feature distribution testing, and empirical evaluation, this document diagnoses all 13 core dimensions of the Q-Nav learning pipeline.

---

## 1. Dataset Quality
- **Current state**: 1,246 samples across 47 trajectories (average 26.5 steps/trajectory).
- **Finding**: The trajectory generation picked random coordinates without guaranteeing diverse compass angles. In individual trajectories, UAV flight paths were dominated by single directional corridors (e.g., predominantly East or North-East), leaving entire quadrants of the state-action space unvisited.
- **Root Cause**: Training environments had random starting and ending points, but lacked systematic angular coverage across all 8 cardinal and intercardinal compass directions.

## 2. Class Imbalance
- **Current state**: In the previous validation split (229 samples), majority class represented **34.93%** of all samples.
- **Finding**: Validation samples had 0% representation for `DOWN_LEFT`, `UP_LEFT`, `UP`, and `DOWN`. Consequently, the classifier collapsed to predicting only `LEFT` (45.4% accuracy) and `RIGHT` (35.0% accuracy), yielding 0.0% accuracy on 5 out of 9 classes.
- **Target**: Balanced representation where each of the 8 directional actions represents 8%–14% of the dataset, and `STAY` represents ~3%–6% (only selected upon goal threshold $\le 2.0$ m or complete obstruction).

## 3. Teacher-Label Quality
- **Current state**: `teacher_policy` evaluated 8 candidate actions with lookahead $K=2$:
  $$\text{score} = (\text{progress} \times 4.0) + \text{align\_score} + \text{clearance\_reward} + \text{smoothness\_penalty}$$
- **Finding**: The teacher policy lacked integration with the Artificial Potential Field (APF) force field. When an obstacle was present at 45°, `progress` and `align_score` could still dominate over clearance unless an outright collision occurred within 2 steps.
- **Resolution**: Expand lookahead from $K=2$ to $K \in \{2, 3, 4, 5\}$ and score candidate actions against the composite guidance force (attractive + repulsive + tangential circulation + boundary) with strict multi-step swept collision verification.

## 4. Feature Separability
- **Finding**: Empirical testing with a non-linear Random Forest on raw 12D features from single-pass trajectories yielded only 44.14% validation accuracy because the 12D features mix body-frame sensor readings (`front_obstacle_dist`, `left_obstacle_dist`, `right_obstacle_dist`, `rel_target_angle`) with world-frame target vectors (`target_dir_x`, `target_dir_y`).
- When systematic multi-sector trajectories were generated covering all 8 compass directions, Random Forest validation accuracy immediately jumped to **74.02%**, proving that feature separability improves drastically with balanced spatial coverage.

## 5. Train / Validation Leakage
- **Audit**: Verified that training and validation datasets are strictly partitioned by trajectory ID (`trajectory_id`).
- **Confirmation**: `train_traj_ids.isdisjoint(val_traj_ids) == True`. No step from any flight path exists in both partitions.
- **Test isolation**: Neither train nor validation sets touch `scenario_01` through `scenario_05` or benchmark obstacle maps (`map_01` through `map_03`).

## 6. Quantum Feature Encoding
- **Current state**: Grouped encoding compresses 12 features into 4 rotation angles ($q_0, q_1, q_2, q_3$):
  - $q_0$: Bearing angle and distance
  - $q_1$: Frontal and nearest obstacle danger
  - $q_2$: Lateral clearance differential (`left_obs - right_obs`)
  - $q_3$: Boundary and density pressure
- **Bottleneck Identified**: For $N=4$ qubits, `target_dir_x` and `target_dir_y` (world target orientation) were omitted from qubits 0–3 and only included on Qubit 4 when $N > 4$. Thus, a 4-qubit model had severe aliasing between global North/South/East/West directions.
- **Resolution**:
  1. For 4-qubit models, ensure $q_0$ explicitly carries the world target orientation $\text{atan2}(t_y, t_x)$ or test 6-qubit and 8-qubit circuits where higher-order angles natively encode target vectors ($q_4$) and kinematics ($q_5$).
  2. Evaluate 4 encoding variants: Grouped, Angle-Scaled, Bounded Nonlinear ($\tanh$), and Grouped-Normalized.

## 7. VQC Expressivity
- **Current state**: 4 qubits, 2 layers, 16 variational parameters ($\boldsymbol{\theta}$).
- **Finding**: A 16-parameter circuit with 2 layers has limited entangling capacity to represent non-linear decision boundaries for 9 discrete classes across complex obstacle configurations.
- **Resolution**: Test higher-capacity architectures across controlled experiments:
  - Model A: 4 qubits, 2 layers (16 params)
  - Model B: 4 qubits, 3 layers (24 params)
  - Model C: 4 qubits, 4 layers (32 params)
  - Model D: 6 qubits, 3 layers (36 params)
  - Model E: 6 qubits, 4 layers (48 params)
  - Model F: 8 qubits, 3 layers (48 params)

## 8. Output-Layer Capacity
- **Current state**: Linear projection $W \in \mathbb{R}^{9 \times 20}$, $\mathbf{b} \in \mathbb{R}^9$.
- **Finding**: A single linear layer mapping 20 quantum features to 9 classes cannot learn non-linear boundaries between adjacent diagonal actions.
- **Resolution**: Evaluate:
  1. Pure linear head
  2. Small regularized MLP head ($20 \to 32 \to 9$)
  3. Hybrid Quantum-Classical classifier fusing quantum features with residual 12D classical features ($[X_{\text{quantum}}, X_{\text{12D}}] \to 32 \to 9$).

## 9. Optimization
- **Current state**: Adam optimizer with fixed learning rates ($\text{lr} = 0.03$, $\text{vqc\_lr} = 0.015$).
- **Resolution**: Evaluate AdamW with weight decay $10^{-4}$, gradient clipping, learning rate scheduler with plateau decay, and controlled learning rates ($0.003, 0.005, 0.01, 0.02$).

## 10. Learning-Rate Behavior
- **Finding**: Finite-difference gradient evaluation for $\boldsymbol{\theta}$ with step size $\epsilon = 0.05$ worked reliably ($\Delta \boldsymbol{\theta} = 1.074$ rad), but classical readout learning rate ($0.03$) caused rapid loss plateauing after epoch 2.
- **Resolution**: Implement learning rate scheduling ($\times 0.7$ on validation loss plateau) to enable fine convergence.

## 11. Early Stopping
- **Current state**: Patience = 5 epochs on validation loss.
- **Finding**: In epoch 2, validation loss reached 2.0368 and stopped improving, terminating at epoch 7.
- **Resolution**: Increase patience to 6–8 epochs to allow VQC parameters to escape shallow local minima before early stopping triggers.

## 12. Checkpoint Selection
- **Finding**: Correctly tracks and saves the best-validation checkpoint (lowest validation loss), restoring optimal weights rather than the terminal epoch state.

## 13. Validation Metric Calculation
- **Finding**: Accuracy, Macro F1, Weighted F1, Per-class Precision/Recall/F1, and Confusion Matrix are mathematically rigorous with zero test leakage.
