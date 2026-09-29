# Aim: Drosophila MaleCNS v1.0 Whole-Brain Connectome Simulation & Escape Reflex

A general framework for instantiating animal brain data into a virtual environment. This project uses the adult male *Drosophila melanogaster* central nervous system connectome dataset (**MaleCNS v1.0**), released by Google and HHMI Janelia, to instantiate and simulate biophysical spiking neural circuits (Leaky Integrate-and-Fire).

---

## 1. Data Sources & Overview

The connectome data is fetched directly from Google Cloud Storage:
* **Base URL**: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

### Downloaded Datasets

| File Name | Description | Source URL |
| :--- | :--- | :--- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | Neuron metadata, classification, soma location, and status labels | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather) |
| `body-neurotransmitters-male-cns-v1.0.feather` | Predicted and consensus neurotransmitters (acetylcholine, GABA, glutamate, etc.) | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather) |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | Synaptic connections and weights (synapse count) between pre- and post-synaptic neurons | [Download](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather) |

---

## 2. Connectome Statistics (MaleCNS v1.0)

Calculated via [`calc_stats.py`](calc_stats.py):

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Total Annotated Bodies** | **`211,577`** | Total bodies present in the annotations table |
| **Traced Neurons** | **`165,122`** | Neurons with `status == 'Traced'` |
| **Traced-to-Traced Connections** | **`25,563,197`** | Directed synaptic connections between traced neurons |
| **Synapses on Traced Connections** | **`124,025,046`** | Total synaptic count carried across traced connections |

---

## 3. Biophysical Model & Berkeley (2024) Parameters

The simulation follows the whole-brain leaky integrate-and-fire (LIF) methodology and biophysical parameter settings established by UC Berkeley researchers:

* **Paper Title**: *A Drosophila computational brain model reveals sensorimotor processing*
* **Authors**: Philip K. Shiu, et al.
* **Journal**: *Nature* **634**, 210–219 (Published: October 2, 2024)
* **Nature Article**: [https://doi.org/10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)
* **bioRxiv Preprint**: [https://doi.org/10.1101/2023.06.02.543426](https://www.biorxiv.org/content/10.1101/2023.06.02.543426v2)
* **Codebase Repository**: [GitHub - philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)

### Leaky Integrate-and-Fire (LIF) Equations

The subthreshold membrane potential $V_i(t)$ of neuron $i$ evolves as:

$$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + I_{\text{syn}, i}(t) + I_{\text{ext}, i}(t)$$

where:
$$I_{\text{syn}, i}(t) = \sum_{j} W_{ij} S_j(t - t_{\text{delay}})$$

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

### Neurotransmitter Sign Assignment

| Neurotransmitter | Classification | Synaptic Sign in $W_{ij}$ |
| :--- | :--- | :--- |
| **Acetylcholine (ACh)** | Excitatory | $+1$ |
| **GABA** | Inhibitory | $-1$ |
| **Glutamate (Glu)** | Inhibitory (predominant in fly CNS) | $-1$ |
| **Others / Unknown** | Default | $+1$ |

---

## 4. Giant Fiber Escape Reflex Circuit Test

The Giant Fiber (GF) circuit is the classic *Drosophila* looming-evoked escape reflex:
1. **Sensory Detection**: Visual looming / optic lobe inputs (PVLP/DNp70) receive external sensory stimulus.
2. **Command Interneuron**: Left and Right Giant Fiber neurons (`DNp01(GF)_L` [ID: 10010] and `DNp01(GF)_R` [ID: 10001]) depolarize and fire at $t = 2.70\text{ ms}$.
3. **Motor Output**: Direct excitation propagates to the tergotrochanteral motor neurons (TTMn) and peripheral synapsing interneurons (PSI) to drive the jump-and-escape sequence.

Run the test:
```bash
python simulate_escape.py
```

---

## 5. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/404kavy/Aim-A-general-framework-for-instantiating-animal-brain-data-into-virtual-environment-.git
cd Aim-A-general-framework-for-instantiating-animal-brain-data-into-virtual-environment-

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download connectome datasets
python download_data.py

# Compute connectome statistics
python calc_stats.py

# Run escape reflex simulation
python simulate_escape.py
```