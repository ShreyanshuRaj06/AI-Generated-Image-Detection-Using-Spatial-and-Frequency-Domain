import re

with open("paper/main.tex", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update preamble to include backgrounds library
old_lib = r"\usetikzlibrary{positioning,arrows.meta,fit,calc,shapes.geometric}"
new_lib = r"\usetikzlibrary{positioning,arrows.meta,fit,calc,shapes.geometric,backgrounds}"
assert old_lib in content, "old_lib not found!"
content = content.replace(old_lib, new_lib)

# 2. Redesign Figure 1
old_fig_pattern = re.compile(r'\\begin\{figure\*\}\[!t\].*?\\label\{fig:architecture\}\s*\\end\{figure\*\}', re.DOTALL)
assert old_fig_pattern.search(content), "Old figure not found!"

new_fig = r"""\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    >=Stealth,
    node distance=0.38cm and 0.6cm,
    font=\footnotesize,
    % Box styles
    io_box/.style={
        draw=gray!70!black, fill=gray!12, rounded corners=4pt,
        minimum width=4.6cm, minimum height=0.72cm, align=center, line width=0.7pt
    },
    spatial_box/.style={
        draw=blue!70!black, fill=blue!6, rounded corners=4pt,
        minimum width=4.8cm, minimum height=0.66cm, align=center, line width=0.65pt
    },
    freq_box/.style={
        draw=orange!80!black, fill=orange!7, rounded corners=4pt,
        minimum width=4.8cm, minimum height=0.66cm, align=center, line width=0.65pt
    },
    fusion_box/.style={
        draw=purple!80!black, fill=purple!7, rounded corners=4pt,
        minimum width=5.2cm, minimum height=0.70cm, align=center, line width=0.75pt
    },
    out_box/.style={
        draw=teal!80!black, fill=teal!10, rounded corners=4pt,
        minimum width=5.2cm, minimum height=0.70cm, align=center, line width=0.85pt
    },
    container/.style={
        draw=#1!50!black, fill=#1!2, dashed, rounded corners=6pt, line width=0.7pt,
        inner sep=0.28cm
    },
    arr/.style={->, thick, draw=gray!80!black},
    edge_pill/.style={font=\scriptsize\bfseries, text=gray!80!black, fill=white, draw=gray!40, rounded corners=2pt, inner sep=1.5pt}
]

% ── Top: Shared Input ────────────────────────────────────────────────────────
\node[io_box] (input) {\textbf{Input RGB Image (No Interpolation)}\\[1pt]$\mathbf{X} \in [0,1]^{B \times 3 \times H \times W}$};

% ── Spatial Stream (Left: x = -3.7cm) ────────────────────────────────────────
\node[spatial_box, below left=0.85cm and 1.3cm of input] (spatial_norm) {
    \textbf{ImageNet Normalization}\\[1pt]
    $\hat{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) \,/\, \boldsymbol{\sigma} \quad \text{(Channel-wise)}$
};
\node[spatial_box, below=0.38cm of spatial_norm] (resnet_stem) {
    \textbf{ResNet-18 Stem \& Early Layers}\\[1pt]
    $\text{conv1} (7{\times}7, 64) \to \text{MaxPool} \to \text{layer1} \to \text{layer2}$
};
\node[spatial_box, below=0.38cm of resnet_stem] (resnet_deep) {
    \textbf{Deep Residual Feature Blocks}\\[1pt]
    $\text{layer3} \to \text{layer4} \quad (512 \text{ feature maps})$
};
\node[spatial_box, below=0.38cm of gap1_ref] (gap1) {
    \textbf{Adaptive Global Avg. Pooling}\\[1pt]
    $\text{AvgPool2D}(1{\times}1) \to \mathbf{z}_s \in \mathbb{R}^{B \times 512}$
};
\node[spatial_box, below=0.38cm of gap1] (sproj) {
    \textbf{Spatial Projection Head}\\[1pt]
    $\text{Linear}(512 \to 256) \to \text{BN} \to \text{ReLU} \to \mathbf{p}_s \in \mathbb{R}^{B \times 256}$
};

% ── Frequency Stream (Right: x = +3.7cm) ──────────────────────────────────────
\node[freq_box, below right=0.85cm and 1.3cm of input] (gray_hann) {
    \textbf{Grayscale \& Hann Window}\\[1pt]
    $Y = 0.299R+0.587G+0.114B \;\odot\; w_{H,W}$
};
\node[freq_box, below=0.38cm of gray_hann] (transform) {
    \textbf{Spectral Decomposition (GPU Matrix)}\\[1pt]
    $\text{Separable 2D-DCT-II} \;\;/\;\; \text{2D-FFT}$
};
\node[freq_box, below=0.38cm of transform] (log_norm) {
    \textbf{Log-Compression \& Normalization}\\[1pt]
    $\mathbf{S} = \text{Norm}\big(\log(1 + |\mathbf{F}|)\big) \in [0,1]^{B \times 1 \times H \times W}$
};
\node[freq_box, below=0.38cm of log_norm] (fcnn) {
    \textbf{4-Stage Spectral CNN Backbone}\\[1pt]
    $\text{Conv}(1 \to 32 \to 64 \to 128 \to 256) \to \text{GAP} \to \mathbf{z}_f \in \mathbb{R}^{B \times 256}$
};
\node[freq_box, below=0.38cm of fcnn] (fproj) {
    \textbf{Frequency Projection Head}\\[1pt]
    $\text{Linear}(256 \to 256) \to \text{BN} \to \text{ReLU} \to \mathbf{p}_f \in \mathbb{R}^{B \times 256}$
};

% ── Bounded Stream Containers with Title Headers ─────────────────────────────
\begin{scope}[on background layer]
\node[container=blue, fit=(spatial_norm) (resnet_stem) (resnet_deep) (gap1) (sproj)] (spatial_group) {};
\node[container=orange, fit=(gray_hann) (transform) (log_norm) (fcnn) (fproj)] (freq_group) {};
\end{scope}

\node[font=\footnotesize\bfseries, text=blue!85!black, fill=blue!10, draw=blue!40, rounded corners=3pt, inner sep=2.5pt, above=0.06cm of spatial_group.north] {
    \textsc{Spatial Feature Stream} (ResNet-18)
};
\node[font=\footnotesize\bfseries, text=orange!90!black, fill=orange!12, draw=orange!45, rounded corners=3pt, inner sep=2.5pt, above=0.06cm of freq_group.north] {
    \textsc{Frequency Spectral Stream} (2D-DCT / 2D-FFT)
};

% ── Bottom: Concatenation, MLP Head & Classifier ─────────────────────────────
\node[fusion_box, below=0.95cm of $(sproj.south)!0.5!(fproj.south)$] (concat) {
    \textbf{Latent Feature Concatenation \& Fusion}\\[1pt]
    $\mathbf{z} = [\mathbf{p}_s \,\|\, \mathbf{p}_f] \in \mathbb{R}^{B \times 512}$
};
\node[fusion_box, below=0.38cm of concat] (mlp) {
    \textbf{Multi-Layer Perceptron (MLP) Classifier Head}\\[1pt]
    $\text{Linear}(512 \to 256) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(256 \to 1)$
};
\node[out_box, below=0.38cm of mlp] (output) {
    \textbf{Decision Probability \& Binary Supervision}\\[1pt]
    $\hat{y} \in \mathbb{R}^{B \times 1} \implies P(\text{AI}) = \sigma(\hat{y}) \quad \Big[\mathcal{L}_{\text{BCEWithLogits}}(\hat{y}, y)\Big]$
};

% ── Connecting Flow Arrows ───────────────────────────────────────────────────
\draw[arr] (input.south) -- ++(0,-0.32) -| (spatial_norm.north);
\draw[arr] (input.south) -- ++(0,-0.32) -| (gray_hann.north);

\draw[arr] (spatial_norm) -- (resnet_stem);
\draw[arr] (resnet_stem) -- (resnet_deep);
\draw[arr] (resnet_deep) -- (gap1);
\draw[arr] (gap1) -- (sproj);

\draw[arr] (gray_hann) -- (transform);
\draw[arr] (transform) -- (log_norm);
\draw[arr] (log_norm) -- (fcnn);
\draw[arr] (fcnn) -- (fproj);

\draw[arr] (sproj.south) -- ++(0,-0.35) -| ([xshift=-1.3cm]concat.north)
    node[pos=0.25, edge_pill] {256-d};
\draw[arr] (fproj.south) -- ++(0,-0.35) -| ([xshift=1.3cm]concat.north)
    node[pos=0.25, edge_pill] {256-d};

\draw[arr] (concat) -- (mlp) node[midway, edge_pill] {512-d};
\draw[arr] (mlp) -- (output) node[midway, edge_pill] {1 logit};

\end{tikzpicture}
\caption{System architecture of the proposed dual-branch spatial-frequency fusion framework for AI-generated image detection. The spatial stream extracts fine-grained spatial and semantic cues through pre-trained ResNet-18 residual blocks, while the frequency stream extracts spectral features directly on GPU buffers via 2D-FFT or orthonormal separable 2D-DCT-II with boundary Hann windowing. Symmetrical projection layers map both modalities into a shared 256-dimensional latent space before feature concatenation and MLP classification.}
\label{fig:architecture}
\end{figure*}"""

# Notice in new_fig line: \node[spatial_box, below=0.38cm of gap1_ref] (gap1)
# should be: \node[spatial_box, below=0.38cm of resnet_deep] (gap1)
new_fig = new_fig.replace("below=0.38cm of gap1_ref", "below=0.38cm of resnet_deep")

content = old_fig_pattern.sub(lambda m: new_fig, content)

# 3. Fix the awkward column break in the DCT text
old_dct_text = r"""and compute the 2D transform as the separable product $\mathbf{C} = \mathbf{D}\,\mathbf{X}\,\mathbf{D}^\mathsf{T}$. The matrix $\mathbf{D}$ is registered as a non-trainable buffer, built once per unique~$N$ and cached."""
new_dct_text = r"""The 2D spectral transformation is computed as the separable matrix product:
\begin{equation}
\label{eq:separable_dct}
\mathbf{C} = \mathbf{D}\,\mathbf{X}\,\mathbf{D}^\mathsf{T}
\end{equation}
where the transform matrix $\mathbf{D}$ is registered as a non-trainable GPU buffer, precomputed once per unique spatial dimension $N$ and cached in memory."""

assert old_dct_text in content, "old_dct_text not found!"
content = content.replace(old_dct_text, new_dct_text)

# 4. Enhance Table I vertical padding
old_table_start = r"\begin{table*}[!t]"
new_table_start = r"""\begin{table*}[!t]
\renewcommand{\arraystretch}{1.18}"""
content = content.replace(old_table_start, new_table_start, 1)

with open("paper/main.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated main.tex successfully!")
