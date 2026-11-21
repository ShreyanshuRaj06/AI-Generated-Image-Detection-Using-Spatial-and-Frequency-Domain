new_tikz = r"""\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    >=Stealth,
    node distance=0.45cm and 0.6cm,
    font=\footnotesize,
    % Box styles
    io_box/.style={
        draw=gray!70, fill=gray!10, rounded corners=4pt,
        minimum width=4.0cm, minimum height=0.75cm, align=center, line width=0.7pt
    },
    spatial_box/.style={
        draw=blue!70!black, fill=blue!7, rounded corners=4pt,
        minimum width=4.4cm, minimum height=0.68cm, align=center, line width=0.6pt
    },
    freq_box/.style={
        draw=orange!80!black, fill=orange!8, rounded corners=4pt,
        minimum width=4.4cm, minimum height=0.68cm, align=center, line width=0.6pt
    },
    fusion_box/.style={
        draw=purple!80!black, fill=purple!8, rounded corners=4pt,
        minimum width=4.8cm, minimum height=0.72cm, align=center, line width=0.7pt
    },
    out_box/.style={
        draw=teal!80!black, fill=teal!10, rounded corners=4pt,
        minimum width=4.8cm, minimum height=0.72cm, align=center, line width=0.8pt
    },
    % Group container style
    container/.style={
        draw=#1!60!black, fill=#1!3, dashed, rounded corners=6pt, line width=0.7pt,
        inner sep=0.35cm
    },
    arr/.style={->, thick, draw=gray!80!black},
    edge_label/.style={font=\scriptsize\bfseries, text=gray!70!black, fill=white, inner sep=1pt}
]

% ── Top: Input ───────────────────────────────────────────────────────────
\node[io_box] (input) {\textbf{Input RGB Image (No Interpolation)}\\[1pt]$\mathbf{X} \in [0,1]^{B \times 3 \times H \times W}$};

% ── Spatial Branch (Left, centered at x = -3.8) ──────────────────────────
\node[spatial_box, below left=0.9cm and 1.6cm of input] (spatial_norm) {
    \textbf{ImageNet Normalization}\\[1pt]
    $\hat{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) \,/\, \boldsymbol{\sigma}$
};
\node[spatial_box, below=0.42cm of spatial_norm] (resnet) {
    \textbf{ResNet-18 Backbone}\\[1pt]
    $\text{conv1} \to \text{layer4 (fine-tuned)}$
};
\node[spatial_box, below=0.42cm of resnet] (gap1) {
    \textbf{Adaptive Global Avg. Pooling}\\[1pt]
    $1 \times 1 \to \mathbf{z}_s \in \mathbb{R}^{B \times 512}$
};
\node[spatial_box, below=0.42cm of gap1] (sproj) {
    \textbf{Projection Layer}\\[1pt]
    $\text{Linear} \to \text{BN} \to \text{ReLU} \to \mathbf{p}_s \in \mathbb{R}^{B \times 256}$
};

% ── Frequency Branch (Right, centered at x = +3.8) ───────────────────────
\node[freq_box, below right=0.9cm and 1.6cm of input] (gray) {
    \textbf{Luminance Grayscale Conversion}\\[1pt]
    $Y = 0.299R + 0.587G + 0.114B$
};
\node[freq_box, below=0.42cm of gray] (hann) {
    \textbf{2D Hann Window Filtering}\\[1pt]
    $\tilde{Y} = Y \odot w \quad \text{(leakage reduction)}$
};
\node[freq_box, below=0.42cm of hann] (transform) {
    \textbf{Spectral Transform (GPU Matrix)}\\[1pt]
    $\text{Separable 2D-DCT-II} \;\;/\;\; \text{2D-FFT} \to \log(1 + |\mathbf{F}|)$
};
\node[freq_box, below=0.42cm of transform] (fcnn) {
    \textbf{4-Stage Spectral CNN Backbone}\\[1pt]
    $1 \to 32 \to 64 \to 128 \to 256 \text{ channels}$
};
\node[freq_box, below=0.42cm of fcnn] (gap2) {
    \textbf{Adaptive Global Avg. Pooling}\\[1pt]
    $1 \times 1 \to \mathbf{z}_f \in \mathbb{R}^{B \times 256}$
};
\node[freq_box, below=0.42cm of gap2] (fproj) {
    \textbf{Projection Layer}\\[1pt]
    $\text{Linear} \to \text{BN} \to \text{ReLU} \to \mathbf{p}_f \in \mathbb{R}^{B \times 256}$
};

% ── Group Containers with Title Badges ────────────────────────────────────
\begin{scope}[on background layer]
\node[container=blue, fit=(spatial_norm) (resnet) (gap1) (sproj)] (spatial_group) {};
\node[container=orange, fit=(gray) (hann) (transform) (fcnn) (gap2) (fproj)] (freq_group) {};
\end{scope}

\node[font=\footnotesize\bfseries, text=blue!80!black, fill=blue!10, rounded corners=2pt, inner sep=3pt, above=0.08cm of spatial_group.north] {
    \textsc{Spatial Domain Feature Stream} (ResNet-18)
};
\node[font=\footnotesize\bfseries, text=orange!90!black, fill=orange!12, rounded corners=2pt, inner sep=3pt, above=0.08cm of freq_group.north] {
    \textsc{Frequency Domain Spectral Stream} (2D-DCT / 2D-FFT)
};

% ── Bottom: Fusion & Classifier (Centered) ────────────────────────────────
\node[fusion_box, below=1.0cm of $(sproj.south)!0.5!(fproj.south)$] (concat) {
    \textbf{Feature Concatenation \& Fusion}\\[1pt]
    $\mathbf{z} = [\mathbf{p}_s \,\|\, \mathbf{p}_f] \in \mathbb{R}^{B \times 512}$
};
\node[fusion_box, below=0.45cm of concat] (mlp) {
    \textbf{Classification MLP Head}\\[1pt]
    $\text{Linear}(512 \to 256) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(256 \to 1)$
};
\node[out_box, below=0.45cm of mlp] (output) {
    \textbf{Sigmoid Activation \& Loss Computation}\\[1pt]
    $\hat{y} \in \mathbb{R}^{B \times 1} \implies P(\text{AI}) = \sigma(\hat{y}) \quad \Big[\mathcal{L}_{\text{BCE}}(\hat{y}, y)\Big]$
};

% ── Connecting Arrows ─────────────────────────────────────────────────────
\draw[arr] (input.south) -- ++(0,-0.35) -| (spatial_norm.north);
\draw[arr] (input.south) -- ++(0,-0.35) -| (gray.north);

\draw[arr] (spatial_norm) -- (resnet);
\draw[arr] (resnet) -- (gap1);
\draw[arr] (gap1) -- (sproj);

\draw[arr] (gray) -- (hann);
\draw[arr] (hann) -- (transform);
\draw[arr] (transform) -- (fcnn);
\draw[arr] (fcnn) -- (gap2);
\draw[arr] (gap2) -- (fproj);

\draw[arr] (sproj.south) -- ++(0,-0.4) -| ([xshift=-1.2cm]concat.north) 
    node[pos=0.25, edge_label] {256-d};
\draw[arr] (fproj.south) -- ++(0,-0.4) -| ([xshift=1.2cm]concat.north)
    node[pos=0.25, edge_label] {256-d};

\draw[arr] (concat) -- (mlp) node[midway, edge_label] {512-d};
\draw[arr] (mlp) -- (output) node[midway, edge_label] {1 logit};

\end{tikzpicture}
\caption{System architecture of the proposed dual-branch spatial-frequency fusion framework for AI-generated image detection. The spatial branch extracts hierarchical spatial features through fine-tuned pre-trained ResNet-18, while the frequency stream extracts spectral features directly on GPU buffers via 2D-FFT or orthonormal separable 2D-DCT-II. Symmetrical projection layers map both modalities into a shared 256-dimensional latent space before concatenation and MLP classification.}
\label{fig:architecture}
\end{figure*}"""

print("TikZ code length:", len(new_tikz))
