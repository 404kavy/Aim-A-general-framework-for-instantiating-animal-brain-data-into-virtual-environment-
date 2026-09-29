# Aim: Embodied Connectome Simulation

> **A general framework for instantiating animal brain data into a virtual environment.**

This project uses the adult male *Drosophila melanogaster* central nervous system connectome dataset (**MaleCNS v1.0**), released by Google and HHMI Janelia, to instantiate and simulate biophysical spiking neural circuits (Leaky Integrate-and-Fire) with real-time sensory-motor embodiment.

---

## 1. Prerequisites & Core Knowledge Requirements

To understand, develop, and contribute to this embodied connectome framework, researchers and developers should be familiar with the following foundational domains:

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    VIRTUAL ENVIRONMENT                      │
   │           (Physics Engine: MuJoCo / Isaac Gym)              │
   └───────────────┬─────────────────────────────▲───────────────┘
     Sensory Input │                             │ Motor Action
     (Photons/Wind)│                             │ (Joint Torques)
                   ▼                             │
   ┌───────────────────────────────┐ ┌───────────┴───────────────┐
   │     SENSORY TRANSDUCTION      │ │     MOTOR TRANSDUCTION    │
   │ • Visual: Ommatidia / EMD     │ │ • Descending Neurons (DNs)│
   │ • Proprioception: Sensilla    │ │ • VNC Motor Neurons (MNs) │
   │ • Antennal: Johnston's organ  │ │ • Muscle Force Activation │
   └───────────────┬───────────────┘ └───────────▲───────────────┘
                   │ Spikes                      │ Spikes
                   ▼                             │
   ┌─────────────────────────────────────────────┴───────────────┐
   │                WHOLE-BRAIN CONNECTOME CORE                  │
   │           (LIF / Sparse Graph Neural Dynamics)              │
   │  • 165,000+ Neurons | 25,000,000+ Synaptic Connections      │
   │  • Neurotransmitter Polarity Mapping (ACh / GABA / Glu)     │
   └─────────────────────────────────────────────────────────────┘
```

### A. Connectomics & Neural Graph Engineering
* **Electron Microscopy (EM) Reconstruction**: High-throughput automated segmentation and proofreading pipelines generating large-scale connectome graphs (FlyWire, MaleCNS v1.0, MANC, Hemibrain).
* **Sparse Graph Theory**: Handling large adjacency matrices ($W_{ij}$) using Compressed Sparse Row (`scipy.sparse.csr_matrix` / PyTorch sparse tensors) representing synaptic counts and connections.
* **Neurotransmitter Sign Assignment**:
  * $\text{Acetylcholine (ACh)} \to \mathbf{+1\ (Excitatory)}$
  * $\text{GABA} \to \mathbf{-1\ (Inhibitory)}$
  * $\text{Glutamate (Glu)} \to \mathbf{-1\ (Predominantly\ Inhibitory\ in\ insect\ CNS)}$

### B. Computational Biophysics & Spiking Neuron Models
* **Leaky Integrate-and-Fire (LIF) Dynamics**: Subthreshold membrane potential integration, decay time constants ($\tau_m$), threshold detection ($V_{\text{th}}$), reset mechanics ($V_{\text{reset}}$), and refractory periods ($t_{\text{ref}}$).
* **Synaptic Latencies**: Modeling transmission delays ($\Delta t \approx 1.8\text{ ms}$) along axonal and dendritic paths.

### C. Biomechanics & Virtual Embodiment (Body Models)
* **Rigid-Body Multibody Physics**: Kinematic hierarchies (head, thorax, abdomen, 6 multi-segmented legs, and wings) and joint constraints in physics engines like **MuJoCo**, **Isaac Gym**, or **PyBullet**.
* **Neuromuscular Junction (NMJ) & Actuation**: Converting motor neuron spike rates into muscle excitation $e(t)$, muscle activation $a(t)$, and joint torques via Hill-type muscle models.

### D. Sensorimotor Transduction & Closed-Loop Control
* **Sensory Transduction**: Translating optical/visual signals (compound eye ommatidia arrays and Elementary Motion Detectors), mechanosensory cues (campaniform sensilla, chordotonal organs), and antennal wind measurements into sensory spike trains.
* **Motor Control Pathways**: Descending command neurons (DNs) coordinating local Ventral Nerve Cord (VNC) Central Pattern Generators (CPGs) for walking gaits and ballistic escape jumps.

### E. Multi-Timescale Simulation Synchronization
* **Neural Sub-stepping**: $\Delta t_{\text{neural}} \approx 0.1\text{ ms} - 1.0\text{ ms}$ ($1\text{ kHz} - 10\text{ kHz}$)
* **Physics Integration**: $\Delta t_{\text{physics}} \approx 0.5\text{ ms} - 2.0\text{ ms}$ ($500\text{ Hz} - 2\text{ kHz}$)
* **Visual Rendering & Environment**: $\Delta t_{\text{env}} \approx 10\text{ ms} - 16.6\text{ ms}$ ($60\text{ Hz} - 100\text{ Hz}$)

---

## 2. Key Foundational Research Papers

1. **Whole-Brain Drosophila Simulation**:
   * *Shiu et al. (Nature, Oct 2024)*: [A Drosophila computational brain model reveals sensorimotor processing](https://doi.org/10.1038/s41586-024-07763-9) | [Code Repository](https://github.com/philshiu/Drosophila_brain_model)
2. **Biomechanical Embodiment Frameworks**:
   * *Lobato-Ríos et al. (Nature Methods, 2022)*: [NeuroMechFly: a physics-based simulation of adult Drosophila melanogaster](https://doi.org/10.1038/s41592-022-01466-7)
   * *Wang-Chen et al. (Nature Methods, 2024)*: NeuroMechFly v2 (Embodied sensory-motor fly modeling in MuJoCo).
3. **Connectome Datasets**:
   * *Dorkenwald et al. (Nature, 2024)*: Neuronal wiring diagram of an adult brain (FlyWire).
   * *Cheong et al. (Janelia / Google, 2024)*: MaleCNS v1.0 connectome.

---

## 3. Data Sources & Overview

The connectome data is fetched directly from Google Cloud Storage:
* **Base URL**: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

### Datasets

| File Name | Description | Source URL |
| :--- | :--- | :--- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | Neuron metadata, classification, soma location, and status labels | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather) |
| `body-neurotransmitters-male-cns-v1.0.feather` | Predicted and consensus neurotransmitters (acetylcholine, GABA, glutamate, etc.) | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather) |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | Synaptic connections and weights (synapse count) between pre- and post-synaptic neurons | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather) |

---

## 4. Connectome Statistics (MaleCNS v1.0)

Calculated via [`calc_stats.py`](calc_stats.py):

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Total Annotated Bodies** | **`211,577`** | Total bodies present in the annotations table |
| **Traced Neurons** | **`165,122`** | Neurons with `status == 'Traced'` |
| **Traced-to-Traced Connections** | **`25,563,197`** | Directed synaptic connections between traced neurons |
| **Synapses on Traced Connections** | **`124,025,046`** | Total synaptic count carried across traced connections |

---

## 5. Biophysical Model & Berkeley (2024) Parameters

The leaky integrate-and-fire (LIF) dynamics are formulated as:

$$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + \sum_{j} W_{ij} S_j(t - t_{\text{delay}}) + I_{\text{ext}, i}(t)$$

When $V_i(t) \ge V_{\text{th}}$, neuron $i$ fires an action potential ($S_i(t) = 1$), resets to $V_{\text{reset}}$, and enters an absolute refractory period $t_{\text{ref}}$.

### Parameter Values

| Parameter | Symbol | Value | Description |
| :--- | :--- | :--- | :--- |
| **Membrane Time Constant** | $\tau_m$ | $20.0\text{ ms}$ | Passive membrane leak time constant |
| **Synaptic Time Constant** | $\tau_s$ | $5.0\text{ ms}$ | Postsynaptic current decay |
| **Refractory Period** | $t_{\text{ref}}$ | $2.2\text{ ms}$ | Post-spike refractory window |
| **Synaptic Latency** | $t_{\text{delay}}$ | $1.8\text{ ms}$ | Axonal transmission delay |
| **Resting Potential** | $V_{\text{rest}}$ | $0.0\text{ (normalized)}$ | Baseline resting voltage |
| **Threshold Potential** | $V_{\text{th}}$ | $1.0\text{ (normalized)}$ | Spiking threshold voltage |
| **Reset Potential** | $V_{\text{reset}}$ | $0.0\text{ (normalized)}$ | Voltage immediately after spike |

---

## 6. Giant Fiber Escape Reflex Circuit Test

The Giant Fiber (GF) circuit is the classic *Drosophila* looming-evoked escape reflex:
1. **Sensory Detection**: Visual looming / optic lobe inputs (PVLP/DNp70) receive external sensory stimulus.
2. **Command Interneuron**: Left and Right Giant Fiber neurons (`DNp01(GF)_L` [ID: 10010] and `DNp01(GF)_R` [ID: 10001]) depolarize and fire at $t = 2.70\text{ ms}$.
3. **Motor Output**: Direct excitation propagates to the tergotrochanteral motor neurons (TTMn) and peripheral synapsing interneurons (PSI) to drive the jump-and-escape sequence.

### Running the Live Interactive OpenCV Simulation
To test the escape reflex with live camera looming detection and real-time LIF oscilloscope:

```bash
python live_escape_sim.py
```
* **Looming Trigger**: Wave or quickly approach your hand towards the camera to simulate a visual looming threat.
* **Oscilloscope**: Shows live voltage integration across Sensory, Giant Fiber, and Motor channels.
* **Exit**: Press `q` or `ESC`.

---

## 7. Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/404kavy/Aim-A-general-framework-for-instantiating-animal-brain-data-into-virtual-environment-.git
cd Aim-A-general-framework-for-instantiating-animal-brain-data-into-virtual-environment-

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download data
python download_data.py

# Run stats calculation
python calc_stats.py

# Run escape reflex test (offline or live GUI)
python simulate_escape.py
python live_escape_sim.py
```