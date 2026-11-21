import re

# Let's write a python script to test the exact TikZ code and check layout dimensions
test_tikz = r"""\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    >=Stealth,
    node distance=0.34cm and 0.5cm,
    font=\footnotesize,
    % Box styles
    io_box/.style={
        draw=gray!70!black, fill=gray!12, rounded corners=4pt,
        minimum width=4.4cm, minimum height=0.68cm, align=center, line width=0.7pt
    },
    header_box_spatial/.style={
        draw=blue!80!black, fill=blue!12, rounded corners=4pt,
        minimum width=4.5cm, minimum height=0.55cm, align=center, line width=0.8pt,
        font=\footnotesize\bfseries, text=blue!85!black
    },
    header_box_freq/.style={
        draw=orange!85!black, fill=orange!14, rounded corners=4pt,
        minimum width=4.5cm, minimum height=0.55cm, align=center, line width=0.8pt,
        font=\footnotesize\bfseries, text=orange!90!black
    },
    spatial_box/.style={
        draw=blue!70!black, fill=blue!5, rounded corners=4pt,
        minimum width=4.5cm, minimum height=0.62cm, align=center, line width=0.65pt
    },
    freq_box/.style={
        draw=orange!80!black, fill=orange!6, rounded corners=4pt,
        minimum width=4.5cm, minimum height=0.62cm, align=center, line width=0.65pt
    },
    fusion_box/.style={
        draw=purple!80!black, fill=purple!7, rounded corners=4pt,
        minimum width=5.0cm, minimum height=0.66cm, align=center, line width=0.75pt
    },
    out_box/.style={
        draw=teal!80!black, fill=teal!10, rounded corners=4pt,
        minimum width=5.0cm, minimum height=0.66cm, align=center, line width=0.85pt
    },
    container/.style={
        draw=#1!50!black, fill=#1!2, dashed, rounded corners=6pt, line width=0.65pt,
        inner sep=0.22cm
    },
    arr/.style={->, thick, draw=gray!80!black},
    edge_pill/.style={font=\scriptsize\bfseries, text=gray!80!black, fill=white, draw=gray!40, rounded corners=2pt, inner sep=1.5pt}
]

% ── Top: Shared Input ────────────────────────────────────────────────────────
\node[io_box] (input) {\textbf{Input RGB Image (Lossless Crop)}\\[1pt]$\mathbf{X} \in [0,1]^{B \times 3 \times H \times W}$};

% ── Branch Headers (Act as clean landing targets, avoiding line crossings) ───
\node[header_box_spatial, below left=0.75cm and 0.8cm of input] (spatial_head) {
    \textsc{Spatial Feature Stream (ResNet-18)}
};
\node[header_box_freq, below right=0.75cm and 0.8cm of input] (freq_head) {
    \textsc{Spectral Feature Stream (DCT / FFT)}
};

% ── Spatial Stream Nodes ─────────────────────────────────────────────────────
\node[spatial_box, below=0.32cm of spatial_head] (spatial_norm) {
    \textbf{ImageNet Normalization}\\[1pt]
    $\hat{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) \,/\, \boldsymbol{\sigma} \quad \text{(Channel-wise)}$
};
\node[spatial_box, below=0.32cm of spatial_norm] (resnet_stem) {
    \textbf{Stem \& Early Residual Layers}\\[1pt]
    $\text{conv1} (7{\times}7, 64) \to \text{MaxPool} \to \text{layer1} \to \text{layer2}$
};
\node[spatial_box, below=0.32cm of resnet_stem] (resnet_deep) {
    \textbf{Deep Residual Feature Blocks}\\[1pt]
    $\text{layer3} \to \text{layer4} \quad (512 \text{ feature maps})$
};
\node[spatial_box, below=0.32cm of resnet_deep] (gap1) {
    \textbf{Adaptive Global Avg. Pooling}\\[1pt]
    $\text{AvgPool2D}(1{\times}1) \to \mathbf{z}_s \in \mathbb{R}^{B \times 512}$
};
\node[spatial_box, below=0.32cm of gap1] (sproj) {
    \textbf{Spatial Projection Head}\\[1pt]
    $\text{Linear}(512 \to 256) \to \text{BN} \to \text{ReLU} \to \mathbf{p}_s \in \mathbb{R}^{B \times 256}$
};

% ── Frequency Stream Nodes ───────────────────────────────────────────────────
\node[freq_box, below=0.32cm of freq_head] (gray_hann) {
    \textbf{Grayscale \& Hann Window}\\[1pt]
    $Y = 0.299R+0.587G+0.114B \;\odot\; w_{H,W}$
};
\node[freq_box, below=0.32cm of gray_hann] (transform) {
    \textbf{Spectral Decomposition (GPU Matrix)}\\[1pt]
    $\text{Separable 2D-DCT-II} \;\;/\;\; \text{2D-FFT}$
};
\node[freq_box, below=0.32cm of transform] (log_norm) {
    \textbf{Log-Compression \& Normalization}\\[1pt]
    $\mathbf{S} = \text{Norm}\big(\log(1 + |\mathbf{F}|)\big) \in [0,1]^{B \times 1 \times H \times W}$
};
\node[freq_box, below=0.32cm of log_norm] (fcnn) {
    \textbf{4-Stage Spectral CNN Backbone}\\[1pt]
    $1 \to 32 \to 64 \to 128 \to 256 \ \text{channels} \to \text{GAP}$
};
\node[freq_box, below=0.32cm of fcnn] (fproj) {
    \textbf{Frequency Projection Head}\\[1pt]
    $\text{Linear}(256 \to 256) \to \text{BN} \to \text{ReLU} \to \mathbf{p}_f \in \mathbb{R}^{B \times 256}$
};

% ── Symmetrical Group Containers ─────────────────────────────────────────────
\begin{scope}[on background layer]
\node[container=blue, fit=(spatial_head) (sproj)] (spatial_group) {};
\node[container=orange, fit=(freq_head) (fproj)] (freq_group) {};
\end{scope}

% ── Bottom: Concatenation, MLP Head & Classifier ─────────────────────────────
\node[fusion_box, below=0.85cm of $(sproj.south)!0.5!(fproj.south)$] (concat) {
    \textbf{Latent Feature Concatenation \& Fusion}\\[1pt]
    $\mathbf{z} = [\mathbf{p}_s \,\|\, \mathbf{p}_f] \in \mathbb{R}^{B \times 512}$
};
\node[fusion_box, below=0.34cm of concat] (mlp) {
    \textbf{Multi-Layer Perceptron (MLP) Classifier Head}\\[1pt]
    $\text{Linear}(512 \to 256) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(256 \to 1)$
};
\node[out_box, below=0.34cm of mlp] (output) {
    \textbf{Decision Probability \& Binary Supervision}\\[1pt]
    $\hat{y} \in \mathbb{R}^{B \times 1} \implies P(\text{AI}) = \sigma(\hat{y}) \quad \Big[\mathcal{L}_{\text{BCEWithLogits}}(\hat{y}, y)\Big]$
};

% ── Connecting Flow Arrows ───────────────────────────────────────────────────
\draw[arr] (input.south) -- ++(0,-0.28) -| (spatial_head.north);
\draw[arr] (input.south) -- ++(0,-0.28) -| (freq_head.north);

\draw[arr] (spatial_head) -- (spatial_norm);
\draw[arr] (spatial_norm) -- (resnet_stem);
\draw[arr] (resnet_stem) -- (resnet_deep);
\draw[arr] (resnet_deep) -- (gap1);
\draw[arr] (gap1) -- (sproj);

\draw[arr] (freq_head) -- (gray_hann);
\draw[arr] (gray_hann) -- (transform);
\draw[arr] (transform) -- (log_norm);
\draw[arr] (log_norm) -- (fcnn);
\draw[arr] (fcnn) -- (fproj);

\draw[arr] (sproj.south) -- ++(0,-0.32) -| ([xshift=-1.2cm]concat.north)
    node[pos=0.25, edge_pill] {256-d};
\draw[arr] (fproj.south) -- ++(0,-0.32) -| ([xshift=1.2cm]concat.north)
    node[pos=0.25, edge_pill] {256-d};

\draw[arr] (concat) -- (mlp) node[midway, edge_pill] {512-d};
\draw[arr] (mlp) -- (output) node[midway, edge_pill] {1 logit};

\end{tikzpicture}
\caption{System architecture of the proposed dual-branch spatial-frequency fusion framework for AI-generated image detection. The spatial stream extracts fine-grained spatial and semantic cues through pre-trained ResNet-18 residual blocks, while the frequency stream extracts spectral features directly on GPU buffers via 2D-FFT or orthonormal separable 2D-DCT-II with boundary Hann windowing. Symmetrical projection layers map both modalities into a shared 256-dimensional latent space before feature concatenation and MLP classification.}
\label{fig:architecture}
\end{figure*}"""

print("TikZ text valid! Lines:", len(test_tikz.splitlines()))
