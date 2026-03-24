# Methodology: Molecular Solubility Prediction Using Graph Neural Networks

## Table of Contents

1. [Research Question and Experimental Design](#1-research-question-and-experimental-design)
2. [Dataset: ESOL](#2-dataset-esol)
3. [Representing Molecules for a Computer](#3-representing-molecules-for-a-computer)
4. [Model Architectures](#4-model-architectures)
5. [Training: How the Models Learn](#5-training-how-the-models-learn)
6. [5-Fold Cross-Validation: Ensuring Reliable Results](#6-5-fold-cross-validation-ensuring-reliable-results)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results](#8-results)
9. [GAT Attention Visualization: Interpreting What the Model Learned](#9-gat-attention-visualization-interpreting-what-the-model-learned)
10. [Program Structure and Schematic Overview](#10-program-structure-and-schematic-overview)

---

## 1. Research Question and Experimental Design

**Research question:** Does preserving the full molecular graph topology (the GNN approach) yield better solubility predictions than compressing molecules into fixed-length fingerprint vectors (the traditional MLP approach)?

To answer this, we designed a **controlled experiment** comparing three neural network architectures on the same dataset, the same data splits, the same number of training iterations, and the same architecture size. The only thing that changes between the three models is *how they read the molecule*:

| Model | What it sees | How it processes neighbours |
|-------|-------------|---------------------------|
| **MLP** (baseline) | A 1024-bit fingerprint — a lossy summary of the molecule | No neighbours; treats the fingerprint as a flat list of numbers |
| **GCN** | The full molecular graph (atoms + bonds) | Aggregates neighbour information using a fixed weighting based on each atom's number of bonds (degree) |
| **GAT** | The full molecular graph (atoms + bonds) | Aggregates neighbour information using *learned* attention weights — the model decides which neighbours matter most |

Everything else is held constant: 2 hidden layers, 128 neurons per layer, ReLU activation, batch normalisation, Adam optimiser (learning rate 0.001), MSE loss function, 10,000 training epochs, and batch size 64. This ensures that any difference in predictive accuracy can be attributed to the representation and layer type, not to differences in model capacity or training procedure.

Model performance is assessed using **5-fold cross-validation** on the training set (explained in detail in Section 6), followed by a final evaluation on a completely held-out test set that no model ever sees during training or validation.

---

## 2. Dataset: ESOL

The **ESOL (Estimated SOLubility)** dataset was created by Delaney in 2004 and is the most widely cited solubility benchmark in machine learning for chemistry. It contains **1,128 organic molecules** with experimentally measured aqueous solubility values expressed as log S (log of molar solubility in mol/L).

- **Target variable:** log S, ranging from −11.6 (extremely insoluble) to +1.58 (highly soluble), with a mean of −3.05 and standard deviation of 2.10.
- **Molecule sizes:** range from 1 to 32 heavy atoms, with a mean of approximately 11 atoms per molecule.
- **Source:** loaded directly from PyTorch Geometric's MoleculeNet collection.

**Why ESOL?** It is small enough to train on a laptop CPU in reasonable time, well-understood in the literature (providing benchmarks to compare against), and directly relevant to drug development — over 40% of drug candidates fail due to poor solubility. It also exercises the key distinction between fingerprint-based and graph-based approaches: molecular topology matters for solubility because the *position* of functional groups (not just their presence) affects how a molecule interacts with water.

**Data split:** The 1,128 molecules are divided using a fixed random seed (42) to ensure reproducibility:
- **Training set:** 902 molecules (approximately 80%) — used for model training and cross-validation
- **Test set:** 114 molecules (approximately 10%) — held out entirely, used only for final evaluation after all model selection is complete

---

## 3. Representing Molecules for a Computer

A central challenge in molecular machine learning is converting a chemical structure into a format that a neural network can process. We use two fundamentally different representations, and the comparison between them is the heart of this project.

### 3.1 Molecular Graphs (used by GCN and GAT)

A molecule is naturally a graph: **atoms are nodes** and **chemical bonds are edges**. For example, ethanol (CH₃CH₂OH) has 3 heavy atoms (2 carbons and 1 oxygen) connected by 2 bonds.

Each atom (node) is described by **9 numerical features** capturing its chemical identity and local environment:

| Feature | What it describes | Example |
|---------|------------------|---------|
| Atomic number | Which element the atom is | Carbon = 6, Oxygen = 8, Nitrogen = 7 |
| Degree | How many bonds the atom has to other heavy atoms | A carbon in a benzene ring has degree 2 |
| Formal charge | The atom's charge | 0 for neutral, +1 or −1 for ions |
| Hybridisation | The geometry of bonds around the atom | sp³ (tetrahedral), sp² (planar), sp (linear) |
| Aromaticity | Whether the atom is part of an aromatic ring | 1 for benzene carbons, 0 otherwise |
| Number of hydrogens | How many hydrogen atoms are attached | Methyl (CH₃) carbon has 3 hydrogens |
| Chirality | The 3D handedness of the atom | R or S stereocentre, or none |
| Ring membership | Whether the atom is in any ring | 1 for cyclohexane carbons, 0 for a chain |
| Radical electrons | Unpaired electrons | 0 for most stable molecules |

Each bond (edge) carries 3 features: bond type (single, double, triple, aromatic), whether it is conjugated, and whether it is in a ring. Note that in our current implementation, the models use only the node features and the connectivity information (which atoms are bonded to which); the edge features are stored in the dataset and available for future extension.

**Key advantage of graphs:** The full spatial arrangement of atoms and bonds is preserved. The model can distinguish between positional isomers (molecules with the same atoms and bonds arranged differently), which is important because, for example, 1-pentanol and 3-pentanol have the same molecular formula but different solubilities.

### 3.2 Morgan Fingerprints (used by MLP)

Morgan fingerprints (also called Extended-Connectivity Fingerprints, ECFP) are the traditional way to represent molecules for machine learning. They compress a molecule into a **fixed-length binary vector** (we use 1024 bits) using the following process:

```
MORGAN FINGERPRINT GENERATION

Input:  a molecule (from its SMILES string), radius = 2, fingerprint length = 1024
Output: a vector of 1024 zeros and ones

1. Parse the SMILES string into a molecular structure using RDKit
2. Start with an empty fingerprint (all 1024 bits set to 0)
3. For each atom in the molecule:
     For each neighbourhood radius d = 0, 1, 2:
       a. Extract the circular substructure centred on this atom
          extending d bonds outward
       b. Compute a hash (a numerical summary) of this substructure
       c. Map the hash to a position in the fingerprint: position = hash mod 1024
       d. Set that bit to 1
4. Return the fingerprint
```

For example, citronellal (C₁₀H₁₈O) produces a fingerprint with 23 out of 1024 bits set to 1 (2.2% density). Bit 650 encodes the presence of an oxygen atom; bit 1004 encodes the C=O aldehyde group.

**Key limitation of fingerprints:** This process is lossy. The spatial arrangement of atoms is discarded — the fingerprint records *which* substructures are present but not *where* they are relative to each other. Hash collisions also occur (different substructures can map to the same bit). These are precisely the limitations that motivate using graph neural networks instead.

---

## 4. Model Architectures

All three models follow the same high-level pattern:

```
INPUT  →  [Layer 1 + Normalise + Activate]  →  [Layer 2 + Normalise + Activate]  →  OUTPUT
```

Specifically: two hidden layers with 128 neurons each, batch normalisation after each layer (which stabilises training by normalising activations), ReLU activation (which introduces non-linearity by setting negative values to zero), and a single output neuron that produces the predicted log S value.

The only variable is what "Layer 1" and "Layer 2" do internally — how they process the input.

### 4.1 MLP — Multi-Layer Perceptron (Baseline)

The MLP is a standard fully connected neural network. It takes the 1024-bit Morgan fingerprint as input and processes it through two dense layers. Every input bit is connected to every neuron in the next layer; there is no awareness of molecular structure.

```
MLP FORWARD PASS

Input: fingerprint vector (1024 numbers)

Step 1 — First hidden layer:
    Multiply the 1024-dimensional input by a learned weight matrix (1024 x 128)
    Add a learned bias
    Normalise activations (batch normalisation)
    Apply ReLU: replace any negative values with zero
    Result: a 128-dimensional vector

Step 2 — Second hidden layer:
    Multiply the 128-dimensional vector by another weight matrix (128 x 128)
    Add a learned bias, normalise, apply ReLU
    Result: another 128-dimensional vector

Step 3 — Output:
    Multiply by a final weight matrix (128 x 1)
    Result: a single number — the predicted log S
```

The MLP serves as the **baseline**. It represents the traditional approach to molecular property prediction: compress the molecule into a fixed-size vector, then feed it to a generic neural network. Any improvement from the GCN or GAT over the MLP tells us how much value the graph structure adds.

### 4.2 GCN — Graph Convolutional Network

The GCN (Kipf & Welling, 2017) operates directly on the molecular graph instead of a fingerprint. Its key innovation is the **graph convolution**: each atom updates its representation by collecting information from its bonded neighbours.

The GCN uses a fixed weighting scheme: when an atom aggregates messages from its neighbours, each neighbour's contribution is weighted by the inverse square root of the product of their degrees. This normalisation prevents high-degree atoms (those with many bonds) from overwhelming the representation.

```
GCN FORWARD PASS

Input: molecular graph
       - atom features: each atom described by 9 numbers
       - bond connectivity: which atoms are bonded to which

Step 1 — First graph convolution:
    For each atom v in the molecule:
        Look at all atoms u bonded to v (its neighbours)
        For each neighbour u:
            Multiply u's feature vector by a learned weight matrix
            Scale by 1 / sqrt(degree(u) * degree(v))
        Sum all these weighted neighbour contributions
        Normalise (batch normalisation) and apply ReLU
    Result: each atom now has a new 128-dimensional representation
            that encodes its own features AND its immediate neighbours

Step 2 — Second graph convolution:
    Repeat the same process on the updated representations
    Result: each atom's representation now encodes information from
            all atoms within 2 bonds (its 2-hop neighbourhood)

Step 3 — Graph-level readout (global mean pooling):
    Average the representations of ALL atoms in the molecule
    into a single 128-dimensional vector representing the whole molecule
    This step is necessary because different molecules have different
    numbers of atoms, but the output layer expects a fixed-size input

Step 4 — Output:
    Pass the molecule-level vector through a linear layer (128 → 1)
    Result: predicted log S
```

**Why global mean pooling?** Molecules have variable numbers of atoms (1 to 32 in ESOL), but the output layer needs a fixed-size input. Mean pooling averages all atom representations into a single vector, and it is invariant to molecule size — a molecule with 30 atoms and one with 5 atoms both produce a single 128-dimensional vector.

**What does "2-hop neighbourhood" mean?** After layer 1, each atom knows about its directly bonded neighbours (1 hop). After layer 2, the information has propagated one step further — each atom now encodes information from neighbours-of-neighbours (2 hops). This means a carbon atom can "see" atoms two bonds away, which roughly corresponds to the range of influence of local chemical effects.

### 4.3 GAT — Graph Attention Network

The GAT (Velickovic et al., 2018) has the same overall structure as the GCN — graph convolutions, pooling, and a linear output — but it replaces the fixed degree-based weighting with **learned attention coefficients**. Instead of treating all neighbours equally (scaled only by degree), the GAT learns to focus on the neighbours that are most important for the prediction.

```
GAT FORWARD PASS

Input: molecular graph (same as GCN)

Step 1 — First graph attention layer (4 attention heads):
    For each atom v in the molecule:
        For each of 4 independent attention heads:
            For each neighbour u of v:
                Compute an attention score:
                  1. Transform both v's and u's features with learned weights
                  2. Concatenate them
                  3. Pass through a small learned scoring function
                  4. Apply LeakyReLU (allows small negative values through)
                This score measures "how relevant is neighbour u to atom v?"
            Normalise scores across all neighbours using softmax
            (so they sum to 1 — now they are attention weights)
            Compute a weighted sum of neighbour features using these weights
        Concatenate the outputs of all 4 heads → 128-dimensional vector
        Normalise (batch normalisation) and apply ReLU

Step 2 — Second graph attention layer:
    Same process, applied to the updated representations
    Now each atom has attended to its 2-hop neighbourhood

Step 3 — Global mean pooling → Step 4 — Linear output
    (Same as GCN)
```

**Why 4 attention heads?** Each head independently learns a different notion of "importance." One head might learn to attend to electronegative atoms (like oxygen), while another might focus on ring membership. Using multiple heads and concatenating their outputs gives the model richer, more diverse representations. With 128 total dimensions split across 4 heads, each head works in 32 dimensions.

**Why attention matters for solubility:** Consider a molecule with both a hydroxyl group (-OH) and a long carbon chain. For predicting solubility, the -OH group is far more important because it forms hydrogen bonds with water. A GCN would weight these neighbours by their degree (number of bonds), which has nothing to do with their chemical relevance to solubility. A GAT can *learn* that the oxygen in -OH matters more for this particular task.

### 4.4 Architecture Comparison Summary

| Component | MLP | GCN | GAT |
|-----------|-----|-----|-----|
| Input | 1024-bit fingerprint | Atom features (9 per atom) | Atom features (9 per atom) |
| Layer type | Fully connected | Graph convolution | Graph attention (4 heads) |
| Layer 1 | 1024 → 128 | 9 → 128 | 9 → 128 (4 heads x 32) |
| Layer 2 | 128 → 128 | 128 → 128 | 128 → 128 (4 heads x 32) |
| Normalisation | Batch normalisation | Batch normalisation | Batch normalisation |
| Activation | ReLU | ReLU | ReLU |
| Readout | None needed | Global mean pooling | Global mean pooling |
| Output | 128 → 1 | 128 → 1 | 128 → 1 |
| How neighbours are weighted | N/A (no graph) | Fixed (by degree) | Learned (attention) |

---

## 5. Training: How the Models Learn

### 5.1 What Training Means

Training a neural network means iteratively adjusting its internal parameters (weights and biases) so that its predictions get closer to the true values. This happens through a loop:

```
TRAINING LOOP (repeated for 10,000 epochs)

For each epoch:
    Shuffle the training data
    Divide it into mini-batches of 64 molecules

    For each mini-batch:
        1. FORWARD PASS: Feed the batch through the model to get predictions
        2. COMPUTE LOSS: Measure how wrong the predictions are using
           Mean Squared Error (MSE) — the average of (predicted - actual)²
        3. BACKPROPAGATION: Calculate how each weight contributed to the error
        4. UPDATE WEIGHTS: Adjust each weight slightly in the direction that
           reduces the error (using the Adam optimiser, learning rate 0.001)

    Record the average training loss for this epoch
    If a validation set is available, evaluate on it (without updating weights)
```

**Why 10,000 epochs?** An epoch means one complete pass through the entire training set. With a small dataset like ESOL (902 training molecules), the model needs many passes to learn the complex relationship between molecular structure and solubility. We use 10,000 epochs to ensure convergence.

**Why mini-batches of 64?** Processing all 902 molecules at once would give very stable but slow weight updates. Processing one molecule at a time would give fast but noisy updates. Batches of 64 strike a balance: reasonably stable gradients with frequent weight updates.

**Why Adam optimiser?** Adam (Adaptive Moment Estimation) adapts the learning rate for each parameter individually based on the history of gradients. This means parameters that are updated infrequently get larger updates, and parameters that are updated frequently get smaller ones. This generally leads to faster, more stable convergence than a simple fixed learning rate.

### 5.2 Graph Batching

For the GCN and GAT, multiple molecular graphs need to be batched together for efficient training. PyTorch Geometric handles this by combining all molecules in a mini-batch into a single large disconnected graph. A separate index vector tracks which atoms belong to which molecule, so that the global mean pooling step correctly averages atom representations per molecule rather than across the entire batch.

### 5.3 Full Training Procedure

The complete training procedure has two phases:

**Phase 1 — Cross-validation (see Section 6 for detailed explanation):**
For each of the three models (MLP, GCN, GAT), 5-fold cross-validation is performed on the 902-molecule training set. This produces robust performance estimates and involves training 5 separate model instances per architecture (15 total training runs).

**Phase 2 — Final model training:**
After cross-validation confirms which approach works best, each model is retrained from scratch on the *entire* 902-molecule training set for 10,000 epochs. These final models are saved and used for the test set evaluation and all subsequent analysis.

---

## 6. 5-Fold Cross-Validation: Ensuring Reliable Results

### 6.1 The Problem Cross-Validation Solves

When training a machine learning model, we need to know how well it will perform on *new, unseen data* — not just the data it was trained on. A model can memorise its training data (this is called **overfitting**) and appear to perform excellently, but then fail on new molecules.

The simplest approach is to split the data into training and test sets, train on one, and evaluate on the other. But with a small dataset like ESOL (1,128 molecules), a single random split can be misleading: the test set might happen to contain mostly "easy" molecules, or it might contain a disproportionate number of outliers. The performance estimate from one split is **unreliable** because it depends heavily on which molecules ended up in which set.

**K-fold cross-validation solves this problem** by systematically rotating which portion of the data is used for evaluation, and then averaging the results. This gives a more robust and trustworthy estimate of model performance.

### 6.2 How 5-Fold Cross-Validation Works

The training set (902 molecules) is randomly divided into 5 non-overlapping groups (folds), each containing approximately 180 molecules. The process then works as follows:

```
5-FOLD CROSS-VALIDATION

Randomly divide the 902 training molecules into 5 equal groups (folds)

Round 1:  Train on folds 2, 3, 4, 5  →  Evaluate on fold 1
Round 2:  Train on folds 1, 3, 4, 5  →  Evaluate on fold 2
Round 3:  Train on folds 1, 2, 4, 5  →  Evaluate on fold 3
Round 4:  Train on folds 1, 2, 3, 5  →  Evaluate on fold 4
Round 5:  Train on folds 1, 2, 3, 4  →  Evaluate on fold 5

Each round:
  - Initialise a completely fresh model (random weights — no memory of previous rounds)
  - Train for 10,000 epochs on the 4 training folds (~720 molecules)
  - Evaluate on the 1 held-out validation fold (~180 molecules)
  - Record RMSE, MAE, and R²

Final result: average and standard deviation of the 5 scores
```

**Every molecule serves as both training data and validation data** across the 5 rounds, but never both at the same time. This means:
- The performance estimate uses *all* 902 training molecules for evaluation (not just a fraction).
- No single lucky or unlucky split can distort the result — the average across 5 rounds smooths out this variability.
- Each round trains a completely fresh model, so there is no information leakage between rounds.

### 6.3 Why 5 Folds?

Five folds is a standard choice that balances two competing concerns:
- **More folds** (e.g., 10) means each validation fold is smaller, so individual fold estimates are noisier, and training takes longer (10 training runs instead of 5).
- **Fewer folds** (e.g., 2 or 3) means larger validation sets but fewer rounds to average over, giving a less stable estimate.

Five folds with 902 molecules gives about 720 training and 180 validation molecules per round — a reasonable balance for our dataset size.

### 6.4 The Held-Out Test Set

Crucially, cross-validation is performed **only on the 902-molecule training set**. The 114-molecule test set is never used during cross-validation. After cross-validation provides performance estimates and confirms that the models are learning, each architecture is retrained from scratch on the full training set. Only then is the test set used — exactly once — for the final, unbiased evaluation.

This two-level approach (CV for model selection and comparison, held-out test for final reporting) is standard practice in machine learning and prevents any form of data leakage.

---

## 7. Evaluation Metrics

Three complementary metrics are used to assess model performance:

### RMSE — Root Mean Squared Error

RMSE measures the typical size of prediction errors in the same units as the target (log S).

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

It squares each error before averaging, which means **large errors are penalised disproportionately**. A model that is mostly accurate but occasionally very wrong will have a high RMSE. Lower is better.

### MAE — Mean Absolute Error

MAE is the simple average of the absolute errors, also in log S units.

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

Unlike RMSE, MAE treats all errors equally regardless of size. Comparing RMSE and MAE together is informative: if RMSE is much larger than MAE, it suggests the model has a few very large outlier errors.

### R² — Coefficient of Determination

R² measures what fraction of the variance in the true solubility values the model explains.

$$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$$

An R² of 1.0 means perfect predictions. An R² of 0.0 means the model is no better than simply predicting the mean solubility for every molecule. Higher is better.

---

## 8. Results

### 8.1 Test Set Evaluation

After 5-fold cross-validation and retraining on the full training set, the final models were evaluated on the 114 held-out test molecules:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| MLP (baseline) | 1.1294 | 0.8545 | 0.6961 |
| GCN | 0.8766 | 0.6948 | 0.8169 |
| **GAT** | **0.7398** | **0.5491** | **0.8696** |

### 8.2 Improvement Over Baseline

Relative to the MLP baseline:

| | GCN improvement | GAT improvement |
|---|---|---|
| RMSE | −22.4% | −34.5% |
| MAE | −18.7% | −35.7% |

Both graph-based models substantially outperform the fingerprint-based MLP, and the GAT outperforms the GCN. This supports the hypothesis that preserving molecular graph structure leads to better solubility predictions, and that learned attention weights (GAT) are more effective than fixed degree-based weighting (GCN).

### 8.3 Loss Curve Analysis

Training and validation loss curves, averaged across the 5 CV folds, are plotted in the training notebook. Key observations:
- The GCN and GAT converge faster and to lower final MSE values than the MLP on both training and validation data.
- The GAT achieves the lowest validation loss floor, consistent with its best test performance.
- The MLP shows slower convergence and a higher loss plateau, consistent with the information bottleneck created by fingerprint compression.

### 8.4 Best and Worst GAT Predictions

The GAT's 5 most accurate predictions on the test set (absolute errors as low as 0.006 log S units) are small to medium-sized molecules with common functional groups well-represented in the training data.

The GAT's worst predictions (absolute errors around 2.0 log S units) are large, structurally complex molecules — such as glycosides and polycyclic structures — that are underrepresented in ESOL's 1,128 training examples. The model tends to under-predict extreme insolubility for these outliers, suggesting that more training data for large, highly insoluble molecules would improve performance.

### 8.5 Residual Analysis

A histogram of prediction residuals (predicted minus actual) for all three models shows that the GAT's errors are most tightly centred around zero, with fewer extreme outliers. The MLP shows the widest error distribution. All three models show approximately symmetric residuals, indicating no systematic bias toward over- or under-prediction.

---

## 9. GAT Attention Visualization: Interpreting What the Model Learned

One of the key advantages of the GAT over the GCN and MLP is **interpretability**. Because the GAT assigns explicit attention weights to each atom-neighbour pair, we can extract these weights after training and visualise which atoms the model considers most important when predicting solubility. This analysis is performed in the `extra-insights.ipynb` notebook.

### 9.1 Methodology

Three molecules are selected from the test set representing different solubility profiles:

| Category | Molecule | log S | Atoms |
|----------|----------|-------|-------|
| Hydrophobic | Polychlorinated biphenyl | −7.42 | 18 |
| Mixed | Triazine with thioether and alkyl groups | −3.04 | 15 |
| Hydrophilic | Crotonaldehyde (C/C=C/C=O) | +0.32 | 5 |

For each molecule, per-atom attention importance is computed using the following procedure:

```
ATTENTION EXTRACTION

Input: a trained GAT model and a single molecular graph

1. Pass the molecule through both GAT layers,
   requesting that the model return its attention weights

2. Extract the layer-2 attention weights
   (Layer 2 captures the most refined information because it
   builds on the already-processed output of layer 1)
   These weights describe, for every bond in the molecule,
   how much attention the target atom paid to the source atom
   Shape: one weight per bond per attention head

3. Average the weights across the 4 attention heads
   to get a single importance score per bond

4. For each atom, sum up all the incoming attention
   (i.e., how much attention was directed toward this atom
   from all of its neighbours)
   This gives a single "importance" score per atom

5. Normalise the scores to a 0–1 scale

6. Colour-code each atom on a blue-to-red spectrum:
   Blue = low attention (the model considers this atom less important)
   Red = high attention (the model focuses on this atom)

7. Draw the molecule with RDKit, highlighting each atom
   with its attention colour
```

### 9.2 Chemical Interpretation

The attention maps reveal chemically meaningful patterns:

- **Hydrophilic molecule** (crotonaldehyde, log S = +0.32): The model places the highest attention on the carbonyl oxygen (C=O). This makes chemical sense — the oxygen is the atom most capable of forming hydrogen bonds with water, which is the primary driver of high aqueous solubility.

- **Hydrophobic molecule** (polychlorinated biphenyl, log S = −7.42): Attention is spread diffusely across the carbon and chlorine atoms. No single atom dominates the prediction, which is consistent with the chemistry: this molecule is uniformly nonpolar, so solubility (or rather insolubility) is a property of the whole structure, not of any single functional group.

- **Mixed molecule** (triazine derivative, log S = −3.04): Attention concentrates on the nitrogen atoms in the aromatic ring and the thioether sulphur, while the alkyl side chains receive much less attention. This aligns with chemical intuition: the heteroatoms (N, S) are polar sites that interact with water, while the alkyl groups are hydrophobic and contribute less to determining the balance of solubility.

These results demonstrate that the GAT has learned chemically meaningful relationships: it focuses on polar, hydrogen-bonding-capable atoms when predicting solubility, without being explicitly told to do so. This interpretability is not available from the MLP (which operates on an opaque fingerprint) or the GCN (whose fixed degree-based weights carry no chemical meaning).

---

## 10. Program Structure and Schematic Overview

### 10.1 File Organisation

The project is organised into modular Python files and Jupyter notebooks:

```
molecular-solubility/
│
├── dataset/
│   └── dataset_util.py           Data loading, train/test splitting, K-fold
│                                  index generation, Morgan fingerprint
│                                  conversion, and evaluation metrics
│                                  (RMSE, MAE, R²)
│
├── nn/
│   └── nn_util.py                MLP model definition and MLP-specific
│                                  training/evaluation helper functions
│
├── gnns/
│   └── gnn_util.py               GCN and GAT model definitions, shared
│                                  graph-model training/evaluation helpers,
│                                  and the GAT attention extraction method
│
├── training.ipynb                 Main experiment notebook: loads data,
│                                  runs 5-fold CV for all 3 models, trains
│                                  final models, evaluates on test set,
│                                  generates all comparison plots, and
│                                  exports results
│
├── extra-insights.ipynb           GAT attention weight visualisation:
│                                  loads saved GAT model, selects example
│                                  molecules, extracts and visualises
│                                  per-atom attention importance
│
├── models/
│   ├── mlp.pt                    Saved MLP weights
│   ├── gcn.pt                    Saved GCN weights
│   ├── gat.pt                    Saved GAT weights
│   └── cv_results.json           5-fold CV metrics and per-epoch loss
│                                  histories for all 3 models
│
├── results.xlsx                   Exported results (2 sheets):
│                                  Sheet 1: CV and test metrics for all models
│                                  Sheet 2: per-molecule test predictions
│
└── data/                          ESOL dataset (auto-downloaded)
```

### 10.2 End-to-End Schematic

The following diagram shows how data flows through the entire experimental pipeline:

```
                    ESOL Dataset
                  1,128 molecules
                  (SMILES + log S)
                        │
                        ▼
              ┌─────────────────────┐
              │   Train/Test Split  │
              │   (seed = 42)       │
              └─────────────────────┘
                   │            │
                   ▼            ▼
            Training Set    Test Set
           902 molecules   114 molecules
                   │        (held out)
                   │            │
          ┌────────┴────────┐   │
          ▼                 ▼   │
   5-Fold Cross-       Fingerprint   │
   Validation          Generation    │
   Index Split         (for MLP)     │
          │                 │   │
          ▼                 ▼   │
   ┌─────────────────────────┐  │
   │   For each model:       │  │
   │   5 rounds of:          │  │
   │     Train on 4 folds    │  │
   │     Validate on 1 fold  │  │
   │     Record metrics      │  │
   └─────────────────────────┘  │
          │                     │
          ▼                     │
   Report CV averages           │
   (RMSE, MAE, R²)             │
          │                     │
          ▼                     │
   ┌─────────────────────────┐  │
   │   Retrain each model    │  │
   │   on full training set  │  │
   │   (10,000 epochs)       │  │
   └─────────────────────────┘  │
          │                     │
          ▼                     ▼
   ┌───────────────────────────────┐
   │   Final Test Evaluation       │
   │   (each model predicts on     │
   │    the 114 held-out molecules │
   │    it has never seen)         │
   └───────────────────────────────┘
          │
          ▼
   ┌───────────────────────────────┐
   │   Analysis & Visualisation    │
   │   - Loss curves               │
   │   - Predicted vs actual plots │
   │   - Metrics bar chart         │
   │   - Residual histogram        │
   │   - GAT attention maps        │
   │   - Best/worst predictions    │
   └───────────────────────────────┘
          │
          ▼
   Saved outputs:
     models/*.pt        (trained weights)
     cv_results.json    (metrics + histories)
     results.xlsx       (predictions + summary)
```

### 10.3 Hyperparameter Summary

For reference, all hyperparameters used across the experiment:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train/test split ratio | ~80/10 | Standard practice; enough test data for reliable evaluation |
| Random seed | 42 | Ensures reproducible splits across all models |
| Cross-validation folds | 5 | Standard balance of stability and computational cost |
| Hidden layer size | 128 | Sufficient capacity for ESOL's complexity |
| Number of hidden layers | 2 | Gives each atom a 2-hop neighbourhood view |
| Activation function | ReLU | Standard, computationally efficient non-linearity |
| Batch normalisation | After each hidden layer | Stabilises training over 10,000 epochs |
| Optimiser | Adam | Adaptive learning rates for faster convergence |
| Learning rate | 0.001 | Standard default for Adam |
| Loss function | MSE | Matches RMSE evaluation metric; penalises large errors |
| Epochs | 10,000 | Ensures convergence on a small dataset |
| Batch size | 64 | Balances gradient stability with update frequency |
| GAT attention heads | 4 | Multiple independent attention mechanisms for richer representations |
| Morgan fingerprint bits | 1024 | Standard length; balances information with dimensionality |
| Morgan fingerprint radius | 2 (ECFP4) | Captures substructures up to 2 bonds from each atom |
