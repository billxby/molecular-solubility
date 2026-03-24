# Molecular Solubility Prediction Using Graph Neural Networks — Project Context

## Team
- **Bill Xu**, **David Haoran Zhao**, **Derek Li**
- Course: Integrative Project in Science, Marianopolis College (CEGEP), Winter 2026
- Section 3, Group 3 (MW schedule)
- **Disciplines:** Chemistry + Computer Science

---

## Project Summary

We are comparing three neural network architectures for predicting aqueous molecular solubility (log S) on the ESOL dataset:

1. **MLP (Multi-Layer Perceptron)** — baseline, operates on Morgan fingerprints (1024-bit vectors)
2. **GCN (Graph Convolutional Network)** — operates on molecular graphs, aggregates neighbors weighted by degree
3. **GAT (Graph Attention Network)** — operates on molecular graphs, learns attention weights over neighbors

All three models use the **same architecture size** for fair comparison: 2 hidden layers × 128 neurons, ReLU activation, same train/test split. The only variable is the layer type.

**Research question:** Does preserving full molecular graph topology (GNN approach) yield better solubility predictions than compressing molecules into fixed-length fingerprint vectors (MLP approach)?

---

## Project History

- **Original proposal:** Sparse Graph Neural Networks for Quantum Error Correction (scored 20/25)
- **Instructor feedback:** Too ambitious — need to learn neural networks AND quantum computing deeply
- **Revised proposal:** Molecular solubility prediction using GNNs on ESOL dataset — simpler domain, well-scoped, existing datasets and benchmarks, preserves the GNN comparison core

---

## Dataset: ESOL (Estimated SOLubility)

- Created by **Delaney, 2004** (J Chem Inf Comput Sci)
- **1,128 organic molecules** with experimentally measured log-solubility values (log S in mol/L)
- Most widely cited solubility benchmark in ML for chemistry
- Built into PyTorch Geometric's MoleculeNet collection (one-line loading)
- **Node features (9 per atom):**
  1. Atomic number (element identity: C=6, O=8, N=7, etc.)
  2. Degree (number of bonds to other heavy atoms)
  3. Formal charge (0 for neutral, ±1 for ions)
  4. Hybridization (sp, sp², sp³)
  5. Aromaticity (is atom in an aromatic ring?)
  6. Number of hydrogens attached
  7. Chirality (R/S stereocenter or none)
  8. Is in ring (part of any ring structure?)
  9. Radical electrons (unpaired electrons)
- **Edge features (3 per bond):**
  1. Bond type (single, double, triple, aromatic)
  2. Conjugated (yes/no)
  3. Is in ring (yes/no)

---

## Chemistry Concepts

### Aqueous Solubility
Aqueous solubility measures how much of a substance dissolves in water. It is governed by intermolecular forces between solute and solvent (water):

- **Hydrogen bonding** (most important): Functional groups like -OH (hydroxyl), -NH₂ (amine), -COOH (carboxylic acid) form hydrogen bonds with water → increases solubility
- **Dipole-dipole interactions**: Polar bonds (C=O, C-N) interact with water's dipole → moderate increase in solubility
- **Van der Waals forces**: Weak forces, dominant in nonpolar hydrocarbons → decreases solubility (hydrophobic)

**Hydrophilic groups** (increase solubility): hydroxyl, amine, carboxylic acid, carbonyl
**Hydrophobic groups** (decrease solubility): long carbon chains, aromatic rings, halogens

**Key insight:** Molecular topology matters. The *position* of functional groups affects solubility (e.g., 1-pentanol vs 3-pentanol have different solubilities despite the same molecular formula). This is exactly what GNNs can capture but fingerprints lose.

---

## Computer Science Concepts

### Acronym Reference
- **MLP** — Multi-Layer Perceptron (fully connected neural network)
- **GCN** — Graph Convolutional Network
- **GAT** — Graph Attention Network
- **GNN** — Graph Neural Network (umbrella term)
- **MPNN** — Message Passing Neural Network (general framework unifying all GNNs)
- **ESOL** — Estimated SOLubility (the dataset)
- **ECFP** — Extended-Connectivity Fingerprint (same as Morgan fingerprint)
- **SMILES** — Simplified Molecular-Input Line-Entry System (text encoding of molecules)
- **ReLU** — Rectified Linear Unit (activation function: max(0, x))
- **RMSE** — Root Mean Squared Error (evaluation metric)
- **RDKit** — open-source chemistry toolkit for Python
- **QSPR** — Quantitative Structure-Property Relationship

### Morgan Fingerprints (ECFP)
Morgan fingerprints encode molecular structure as a fixed-length bit vector (we use 1024 bits):

1. For each atom, examine its neighborhood within radius r hops (we use r=2, giving ECFP4)
2. Hash each circular substructure pattern to an index in the 1024-bit vector
3. Flip that bit to 1

**Example — Citronellal (C₁₀H₁₈O, SMILES: `CC(CC=O)CCC=C(C)C`):**
- 11 heavy atoms, 10 bonds
- Morgan fingerprint: 23 bits ON out of 1024 (2.2% density)
- Bits ON at indices: [1, 33, 80, 133, 232, 233, 283, 401, 479, 509, 550, 556, 558, 599, 650, 694, 737, 739, 759, 763, 807, 1004, 1017]
- Bit 650: oxygen atom identity (radius 0)
- Bit 1004: C=O aldehyde group (oxygen + neighbor, radius 1)
- Bit 33: methyl carbon atoms (radius 0) — multiple atoms hash to same bit

**Problems with fingerprints:**
- **Lossy compression**: spatial arrangement lost
- **Hash collisions**: different substructures can map to same bit
- **No global topology**: can't distinguish isomers that differ only in functional group position

### GNN Message Passing
All GNNs follow the same pattern (the MPNN framework):
1. **Message**: each atom looks at its neighbors and computes a message from each
2. **Aggregate**: combine all incoming messages (sum, mean, max)
3. **Update**: use aggregated messages to update the atom's representation

This is repeated for each layer (we use 2 layers = 2 rounds of message passing = each atom sees its 2-hop neighborhood).

### GCN Update Rule
$$h_v^{(l+1)} = \sigma\left(\sum_{u \in N(v)} \frac{1}{\sqrt{d_u \cdot d_v}} \mathbf{W} h_u^{(l)}\right)$$
- Weights neighbors by inverse of degree product
- All neighbors contribute equally (given same degree)

### GAT Update Rule
$$h_v^{(l+1)} = \sigma\left(\sum_{u \in N(v)} \alpha_{vu} \mathbf{W} h_u^{(l)}\right)$$
- $\alpha_{vu}$ are learned attention coefficients
- Network learns which neighbors matter more for the prediction
- **Interpretability**: attention weights can be visualized to see which atoms the model focuses on

### Model Architecture (Unified for Fair Comparison)

| Component | MLP | GCN | GAT |
|-----------|-----|-----|-----|
| Input | 1024 (fingerprint) | 9 (node features) | 9 (node features) |
| Hidden Layer 1 | Linear(128) + ReLU | GCNConv(128) + ReLU | GATConv(128) + ReLU |
| Hidden Layer 2 | Linear(128) + ReLU | GCNConv(128) + ReLU | GATConv(128) + ReLU |
| Readout | Linear(1) | global mean pool → Linear(1) | global mean pool → Linear(1) |
| Output | log S | log S | log S |

Same depth (2 hidden layers), same width (128), same activation (ReLU), same output head. Only the layer type changes.

---

## Tech Stack
- Python 3.x
- PyTorch
- PyTorch Geometric (graph data loading, GCN/GAT layers)
- RDKit (Morgan fingerprint generation, SMILES parsing)
- Matplotlib / Seaborn (visualization)
- GitHub (version control)
- No GPU required — ESOL is small (1,128 molecules), trains in minutes on CPU

---

## Work Division
- **Bill**: MLP baseline, GAT implementation, project coordination
- **David**: GCN implementation, evaluation metrics, results analysis
- **Derek**: Data preprocessing (RDKit fingerprints + PyTorch Geometric graphs), attention visualization

---

## Project Schedule

| Week | Dates | Milestone |
|------|-------|-----------|
| W4-5 | Feb 10–21 | Environment setup, literature review, first instructor meeting |
| W6 | Feb 24–26 | **Developed proposal + oral presentation DUE** |
| W7 | Mar 2–5 | March break — begin model implementation |
| W8-9 | Mar 10–21 | Build all three models, initial training |
| W10 | Mar 24–26 | **First submission of final project + presentation DUE** |
| W11-13 | Mar 31–Apr 14 | Analysis, revision, peer review, second instructor meeting |
| W14-15 | Apr 21–30 | Final revisions, **Final submission DUE Apr 30** |

---

## Bibliography (Vancouver Format)

### Annotated Sources (5)
1. Delaney JS. ESOL: Estimating aqueous solubility directly from molecular structure. J Chem Inf Comput Sci. 2004;44(3):1000–5.
2. Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. In: Proceedings of ICLR; 2017.
3. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. In: Proceedings of ICLR; 2018.
4. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for quantum chemistry. In: Proceedings of ICML; 2017. p. 1263–72.
5. Jiang D, Wu Z, Hsieh CY, Chen G, Liao B, Wang Z, et al. Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. J Cheminform. 2021;13(1):12.

### Additional Sources
6. Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50(5):742–54.
7. Iqbal J, Rehman MU, Tayara H, Chong KT. Attention-based graph neural network for molecular solubility prediction. ACS Omega. 2023;8(3):3024–30.
8. Sanchez-Lengeling B, Reif E, Pearce A, Wiltschko AB. A gentle introduction to graph neural networks. Distill. 2021.

### Source Descriptions
- **(1) Delaney 2004**: Created the ESOL dataset. Used linear regression with 9 molecular descriptors. Established the benchmark we're using.
- **(2) Kipf & Welling 2017**: Introduced GCN. Spectral graph convolutions simplified to first-order approximation. Degree-weighted neighbor aggregation.
- **(3) Veličković et al. 2018**: Introduced GAT. Multi-head attention mechanism for learning which neighbors matter. Interpretable via attention weight visualization.
- **(4) Gilmer et al. 2017**: Introduced the MPNN framework. Showed GCN, GAT, and other GNNs are all special cases of message passing. Applied to quantum chemistry on QM9 dataset.
- **(5) Jiang et al. 2021**: Systematic comparison of descriptor-based (fingerprint) vs graph-based models for molecular property prediction. Directly relevant to our comparison.
- **(6) Rogers & Hahn 2010**: Original paper describing ECFP/Morgan fingerprint algorithm in detail. The fingerprint method we use for the MLP baseline.
- **(7) Iqbal et al. 2023**: Applied attention-based GNNs specifically to solubility prediction. AttentiveFP outperformed other architectures.
- **(8) Sanchez-Lengeling et al. 2021**: Accessible introduction to GNNs with interactive visualizations. Good pedagogical reference.

---

## Key Arguments for the Presentation/Report

1. **Why solubility matters**: Drug development (>40% of drug candidates fail due to poor solubility), environmental chemistry (pollutant transport), materials science
2. **Why GNNs over fingerprints**: Fingerprints lose spatial arrangement, suffer hash collisions, and have no global topology. GNNs preserve full molecular graph structure through message passing.
3. **Why GAT is expected to outperform GCN**: Attention lets the model learn which neighbors matter more (e.g., the -OH group matters more than a CH₃ for solubility). GCN treats all neighbors equally weighted by degree.
4. **Why controlled comparison matters**: Same architecture size (2×128), same dataset, same split, same metric — the only variable is the layer type.
5. **Addressing instructor feedback**: We narrowed scope from quantum error correction to molecular solubility, reducing domain complexity while preserving the GNN comparison core.

---

## Progress Demonstrated (as of Feb 2026)
- ✓ Scope refined from QEC to ESOL (per instructor feedback)
- ✓ Development environments set up on all 3 members' machines
- ✓ ESOL dataset loaded and explored (confirmed 1,128 molecules, 9 node features, 3 edge features)
- ✓ Morgan fingerprints successfully generated from SMILES via RDKit
- ✓ Simple feedforward neural network implemented by hand in Jupyter notebook
- ✓ Literature review complete (8 scholarly sources)

---

## Expected Results
Based on literature (Jiang et al. 2021, Iqbal et al. 2023):
- **MLP** (fingerprint): highest RMSE (worst)
- **GCN** (graph, degree-weighted): moderate RMSE
- **GAT** (graph, attention): lowest RMSE (best)

Evaluation metric: **RMSE** (Root Mean Squared Error) on held-out test set of log S values.

Additional analysis: Visualize GAT attention weights on example molecules to show which atoms the model focuses on, and relate those to known chemistry (e.g., does the model attend to hydrogen-bonding functional groups?).
