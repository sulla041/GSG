# DAC 2027 Abstract Submission Package
**Title**: Beyond Message Passing: Pairwise Interaction Dynamics and Hierarchical Meter Networks for Comprehensive Circuit Timing and PPA Prediction  
**Authors**: Yihan Yi, et al.  
**Track**: 2. Design Automation / AI and Machine Learning for CAD (Timing, Power, and Physical Design)  
**Keywords**: Deep Learning for EDA, Static Timing Analysis, Circuit Representation Learning, Graph Neural Networks, Hierarchical Graph Reduction, Physical Design PPA

---

### Abstract (Plaintext for Submission Portal, ~250 words)

Accurate and fast pre-layout/post-layout PPA (Power, Performance, Area) prediction is pivotal for agile VLSI design. Although graph neural networks (GNNs) have shown promise, conventional Message Passing Neural Networks (MPNNs) face three fundamental roadblocks in circuit modeling: (1) edges are treated as passive binary links rather than continuous physical impedance entities; (2) node-centric feature propagation induces severe over-smoothing and over-squashing across deep logic paths ($>50$ stages); and (3) global graph-level pooling collapses into degree explosion ($O(N)$) and attention dilution. 

In this work, we propose a physics-isomorphic circuit modeling paradigm anchored in three core pillars:
First, we formulate a **Native Relation-Centric Representation** that elevates copper interconnects into continuous physical impedance channels (incorporating wire resistance, parasitic capacitance, and driver fan-out load). 
Second, we abandon classical message passing and formulate the netlist as an interacting particle set governed by **Pairwise Displacement Dynamics**. By accumulating unconstrained continuous spatial/temporal displacements along Eulerian manifolds, our model penetrates ultra-deep causal logic cones ($T \ge 64$) with zero feature degradation. 
Third, we reformulate global graph aggregation as a trainable, depth-balanced embedding process via **Hierarchical Meter Networks** ($K=32$ branching pyramid), achieving strict $O(1)$ node degree and $O(\log N)$ logarithmic reduction while naturally aligning with GPU warp-level SIMT execution.

Extensive evaluations on over 1,500 real post-layout industrial circuits demonstrate the overwhelming superiority of our framework across seven benchmark tracks: achieving $r = 0.9541$ on blind-state arrival time (27.3ps MAE with Kron reduction), $4.02\text{ps}$ MAE on transition slew (44.2% error reduction over AAAI 2024 SOTA), $r = 0.9630$ on cell arc delay via sibling-pin clusters, and near-perfect global PPA predictions ($R^2 = 0.9497$ on total area, $R^2 = 0.8400$ on total leakage power).
