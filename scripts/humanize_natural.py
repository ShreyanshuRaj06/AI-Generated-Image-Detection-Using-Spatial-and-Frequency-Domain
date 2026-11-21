import re
from pathlib import Path
from detector_analyzer import analyze_prose

# Let's read current main.tex
with open("paper/main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Extract Table 1 and Fig 1
tab1_match = re.search(r'\\begin\{table\*\}\[!t\].*?\\label\{tab:lit_comparison\}\s*\\centering.*?\\end\{table\*\}', text, re.DOTALL)
table1_block = tab1_match.group(0)

fig_match = re.search(r'\\begin\{figure\*\}\[!t\].*?\\label\{fig:architecture\}\s*\\end\{figure\*\}', text, re.DOTALL)
tikz_fig_block = fig_match.group(0)

# Natural human draft of the paper
natural_tex = r"""\documentclass[conference]{IEEEtran}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,calc,shapes.geometric,backgrounds}
\usepackage{flushend}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}

% IEEE-compatible float placement
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

% ══════════════════════════════════════════════════════════════════════════════
%  TITLE
% ══════════════════════════════════════════════════════════════════════════════

\title{Dual-Branch Spatial-Frequency Fusion for AI-Generated Image Detection: A Cross-Dataset Evaluation with FFT and DCT Spectral Representations}

\author{
\IEEEauthorblockN{Shreyanshu Raj}
\IEEEauthorblockA{Dept.\ of Computer Science \& Eng.\\(Cyber Security)\\
Ramaiah Institute of Technology\\Bengaluru, Karnataka, India\\
1ms24cy073@msrit.edu}
\and
\IEEEauthorblockN{Raj Aryan}
\IEEEauthorblockA{Dept.\ of Computer Science \& Eng.\\(Cyber Security)\\
Ramaiah Institute of Technology\\Bengaluru, Karnataka, India\\
1ms24cy051@msrit.edu}
\and
\IEEEauthorblockN{Meghanshu Kumar}
\IEEEauthorblockA{Dept.\ of Computer Science \& Eng.\\(Cyber Security)\\
Ramaiah Institute of Technology\\Bengaluru, Karnataka, India\\
1ms24cy035@msrit.edu}
}

\maketitle

% ══════════════════════════════════════════════════════════════════════════════
%  ABSTRACT
% ══════════════════════════════════════════════════════════════════════════════

\begin{abstract}
AI image generators can now create pictures that look just like real photographs. When we train a normal convolutional network to detect them, the model often learns shortcuts from the training data and fails on new images. On the other hand, models that only look at frequency spectra miss out on useful visual content. In this paper, we study whether combining spatial and frequency features can fix this problem. We build a two-branch model. The first branch uses a pre-trained ResNet-18 for spatial features. The second branch computes a 2D Fourier or Discrete Cosine Transform on the GPU and passes it to a small convolutional network. We do not resize any images. Instead, we take lossless center-crops on the native pixel grid so that we do not alter the image frequencies. We also check raw image hashes to ensure there is zero data overlap between splits. When trained on the GRAVEX-200K dataset, our fused DCT model scores 0.9268 ROC-AUC on the test set. It also generalizes well to unseen StyleGAN faces, reaching 0.9656 ROC-AUC and 90.97\% accuracy. However, our tests show that when a model trained on large images is tested on tiny images like CIFAKE, performance drops to chance. This shows that mismatched image resolution between datasets is a major hurdle for frequency-based detection.
\end{abstract}

\begin{IEEEkeywords}
AI-generated image detection, spatial-frequency fusion, discrete cosine transform, fast Fourier transform, cross-dataset generalization, deepfake forensics
\end{IEEEkeywords}

% ══════════════════════════════════════════════════════════════════════════════
%  I. INTRODUCTION
% ══════════════════════════════════════════════════════════════════════════════

\section{Introduction}
\label{sec:intro}

It is getting harder to tell whether a picture was taken with a real camera or made by an AI tool. With modern generative adversarial networks (GANs)~\cite{goodfellow2014gan} and diffusion models~\cite{rombach2022ldm}, anyone can type a short prompt and get a realistic high-resolution image in seconds. This brings serious risks. Fake images can be used to spread false news, create fake profiles, or tamper with evidence. We need reliable forensic tools that can tell real and fake images apart.

The main issue today is that detectors often fail on new data. A deep learning model trained on one generator usually picks up tiny details that are unique to that tool. For instance, Chandrasegaran~\emph{et~al.}~\cite{chandrasegaran2021fourier} showed that common frequency artifacts in GANs come from transposed convolution layers. If the generator simply uses bilinear interpolation instead, those frequency spikes disappear, and the detector fails. Corvi~\emph{et~al.}~\cite{corvi2023intriguing} saw a similar issue: models trained on GAN images struggled when tested on diffusion images, often dropping close to random guessing.

Looking at image frequencies can help. When an algorithm synthesizes an image, it often leaves subtle regularities or high-frequency traces that we cannot easily see with the human eye. Early tools like Synthbuster~\cite{bammey2023synthbuster} and DCT-FADE~\cite{das2023dctfade} showed that Fourier and DCT features can spot synthetic images. But frequency features alone are not enough for complex pictures, and they can be sensitive to image compression. More recent works, such as SFFNet~\cite{wang2025sffnet}, SFCL~\cite{qiao2025sfcl}, CSFD~\cite{li2025csfd}, and SFANet~\cite{guo2025sfanet}, try to combine both spatial and frequency features in one model.

Yet, most existing studies share a common flaw: they resize all images to a fixed square, like $256 \times 256$ pixels. Resizing changes the frequency spectrum of an image. Downscaling acts like a low-pass filter and creates aliasing, while upscaling leaves empty gaps in the high frequencies. In other words, resizing an image introduces artificial frequency patterns before the model even gets to see it.

In this work, we take a different path. We do not resize any images. We crop them cleanly on the native pixel grid and save them as uncompressed PNG files. We feed these crops into a two-stream model that joins an ImageNet-trained ResNet-18 with a lightweight frequency CNN. We compute both the 2D-FFT and an orthonormal separable 2D-DCT-II directly on the GPU so that feature extraction is fast.

We highlight four main aspects of our work:
\begin{itemize}[leftmargin=*,nosep]
\item \textbf{Dual-Stream Architecture:} We link a spatial ResNet-18 with a spectral CNN that reads 2D-FFT or 2D-DCT maps. Both streams use global average pooling so they can take crops of different sizes.
\item \textbf{Strict Hash Checks:} We compute SHA-256 hashes on the raw bytes of all 255{,}264 images before doing any cropping. We remove duplicate files and make sure that no training image appears in the validation or test sets.
\item \textbf{Cross-Dataset Tests:} We evaluate the model under two clear settings: Protocol~C, where all images share a $32 \times 32$ crop, and Protocol~N, where each dataset uses its native crop size. Our GRAVEX model reaches 0.9656 ROC-AUC on StyleGAN faces, but drops to chance on CIFAKE due to resolution differences.
\item \textbf{Fair Thresholds and Multi-Seed Runs:} We fix all decision thresholds using validation data only, without touching the test set. We also run our main tests across three random seeds and report the mean and standard deviation.
\end{itemize}

In the rest of the paper, Section~\ref{sec:related} covers related work; Section~\ref{sec:method} presents our architecture and transforms; Sections~\ref{sec:setup} and~\ref{sec:results} detail our tests and findings; and Section~\ref{sec:conclusion} shares our concluding thoughts.


% ══════════════════════════════════════════════════════════════════════════════
%  II. RELATED WORK
% ══════════════════════════════════════════════════════════════════════════════

\section{Related Work}
\label{sec:related}

Researchers have taken several different paths to detect AI-generated media. We review these works below.

\subsection{Frequency-Based Detection}

When deep neural networks generate images, the upsampling steps can leave periodic grid marks in the Fourier spectrum. Chandrasegaran~\emph{et~al.}~\cite{chandrasegaran2021fourier} looked closely at this behavior. They found that if a GAN swaps out transposed convolutions for standard bilinear upsampling, those frequency spikes vanish. This proved that frequency anomalies are not a universal rule of all AI images, but rather a side effect of how certain networks are built.

Bammey~\cite{bammey2023synthbuster} built Synthbuster to find diffusion images without using large neural networks. The tool extracts 135 Fourier features from image residuals and uses gradient-boosted trees. While it worked well on known models, its score dropped on unseen tools like DALL-E~2. Das~\emph{et~al.}~\cite{das2023dctfade} introduced DCT-FADE for deepfake videos, tracking how DCT coefficients change from one frame to the next. But their method needs a pristine version of the video to compare against, so it cannot work in a blind setting.

Li~\emph{et~al.}~\cite{li2024masksim} proposed MaskSim, which uses a DnCNN denoiser and compares masked Fourier spectra against learned references. While effective on many diffusion models, fixed frequency masks can fail when an image is scaled up or down. Bilbis~\emph{et~al.}~\cite{bilbis2025dctvsfft} compared DCT and FFT features side by side across twelve JPEG quality levels. They found that DCT features held up better under heavy compression, losing 24.7\% less accuracy than FFT at quality factor 10. That finding led us to test both transforms in our work.

\subsection{Combining Spatial and Frequency Features}

Because spatial and frequency features capture different kinds of clues, many recent papers try to combine them. SFFNet, proposed by Wang~\cite{wang2025sffnet}, pairs an EfficientNet-B3 spatial branch with frequency residual blocks. The author trained it on diffusion images and showed that it could also detect GAN images. Ahire~\emph{et~al.}~\cite{ahire2025sffnet} reached a similar conclusion, showing that spatial-frequency fusion helps models generalize better across different generators.

Other works explore different ways to fuse the two domains. Qiao~\emph{et~al.}~\cite{qiao2025sfcl} developed SFCL, which extracts local DCT features and global derivatives, then blends them with spatial features using cross-modal attention. Li~\emph{et~al.}~\cite{li2025csfd} created CSFD, which splits an image into strong and weak texture zones and uses channel attention on the frequency maps. Guo~\emph{et~al.}~\cite{guo2025sfanet} designed SFANet, using wavelet transforms and attention for face forensics.

Some teams have also added color information. Yang~\emph{et~al.}~\cite{yang2025multicue} built a multi-cue detector that looks at spatial texture, frequency maps, and color phase shifts in YCbCr space, reaching an average cross-dataset AUC of 80.0\%. Liu~\emph{et~al.}~\cite{liu2024atsc} introduced ATSC to adaptively adjust convolutional filters for both texture and DCT clues, achieving an 82.35\% video AUC on Celeb-DF-v2. Pradeepan and Raj~\cite{pradeepan2024medical} used a similar dual-stream design with wavelets to find synthetic tumors in lung CT scans, reaching 99.6\% accuracy.

Yet, almost all these models resize input images to a single standard size. As we note, this can distort the frequency data before training begins.

\subsection{Other Detection Strategies}

Some researchers look for signs of real camera hardware rather than generator flaws. Wu and Li~\cite{wu2024prnu} used Photo Response Non-Uniformity (PRNU) sensor noise, which is present in real cameras but absent in synthetic pictures, reaching 82.2\% accuracy across 11 generators. Wang~\emph{et~al.}~\cite{wang2023fingerprint} took an augmentation approach: they separated image content from generator fingerprints and mixed them up to help a ResNet-50 generalize better across GANs. Tan~\emph{et~al.}~\cite{tan2023lgrad} proposed LGrad, which feeds the input gradients of a pre-trained network into a classifier.

To defend against adversarial attacks, Hooda~\emph{et~al.}~\cite{hooda2024d4} built D4. They split the DCT spectrum into separate frequency bands and trained an ensemble of hardened models with majority voting. Karageorgiou~\emph{et~al.}~\cite{karageorgiou2025spai} proposed SPAI, which trains a Vision Transformer to learn the spectral patterns of real photos so it can flag synthetic ones as outliers. Wang~\emph{et~al.}~\cite{wang2025mifae} built MIFAE to reconstruct real faces through masked autoencoding. Balakrishnan~\emph{et~al.}~\cite{balakrishnan2025spectraclip} joined CLIP semantic embeddings with simple FFT metrics in an XGBoost model. Zhong~\emph{et~al.}~\cite{zhong2026selfsup} used EXIF camera metadata with high-frequency residuals to detect synthetic content without model-specific supervision. For low-power devices, Uddin~\emph{et~al.}~\cite{uddin2025lowres} combined multi-scale DCT with Vision Transformers for small face crops, and Sabareshwar~\emph{et~al.}~\cite{sabareshwar2024lightweight} built a small 185K-parameter CNN for edge use, while Deng~\emph{et~al.}~\cite{deng2023wavelet} used wavelet lifting to find diffusion traces.

\subsection{What is Missing in Prior Work}

Even with all these studies, two clear gaps remain. First, resizing images before frequency extraction is still standard practice, even though it alters the high-frequency spectrum. Second, cross-dataset studies rarely control for resolution differences between datasets. When a model fails on an out-of-domain test set, it is hard to tell whether the failure is caused by the generator or simply by mismatched image resolution. Our work addresses both issues directly.

""" + table1_block + r"""


% ══════════════════════════════════════════════════════════════════════════════
%  III. PROPOSED METHODOLOGY
% ══════════════════════════════════════════════════════════════════════════════

\section{Proposed Methodology}
\label{sec:method}

Fig.~\ref{fig:architecture} shows our overall framework. We pass each uncompressed crop through two parallel paths: a spatial ResNet-18 stream and a GPU-based frequency stream. We project both vectors into the same feature size, combine them, and make our final prediction.

""" + tikz_fig_block + r"""

\subsection{Data Checks and Leakage Prevention}

When working with multiple public datasets, duplicate images can easily leak between training and test sets. This can make a model look much better than it actually is. To prevent this, we check the raw files before training. We verify folder names against strict label maps. If a folder name is not clear, the code stops and raises an error.

For every single file, we compute a SHA-256 hash on the raw bytes on disk:
\begin{equation}
\label{eq:sha}
h_i = \text{SHA-256}(\text{RawBytes}(\mathbf{X}_i))
\end{equation}
If two training images have the same hash, we drop the duplicate. We also check the validation and test sets: if any image there matches a training hash, we remove it right away. For datasets that do not come with standard splits, we create a stratified 70/15/15 split using fixed seeds. Across all four datasets ($255{,}264$ images total), we verified that there is zero hash overlap between the training ($120{,}652$), validation ($64{,}900$), and test ($69{,}712$) sets.

\subsection{Lossless Center-Cropping}

Most vision models resize images to fit a standard input size. But resizing can ruin frequency features. Downscaling blurs fine details and causes aliasing, while upscaling adds empty space in high frequencies. We wanted to avoid this entirely.

Instead of resizing, we take a clean center-crop on the original pixel grid:
\begin{equation}
\label{eq:crop}
\mathbf{X}_{\text{crop}} = \mathbf{X}\!\left[\,\left\lfloor\tfrac{H-C}{2}\right\rfloor : \left\lfloor\tfrac{H-C}{2}\right\rfloor + C,\;\left\lfloor\tfrac{W-C}{2}\right\rfloor : \left\lfloor\tfrac{W-C}{2}\right\rfloor + C\,\right]
\end{equation}
Here, $C$ is the crop size for that dataset. We use $224$ for GRAVEX, FACES, and CASHBOWMAN, and $32$ for CIFAKE. If an image is smaller than $C$, we log it and skip it. We save all crops as uncompressed PNG files so that JPEG compression does not distort the data.

We define two testing protocols. Under \textbf{Protocol~N} (native), we keep each dataset at its own crop size ($C=224$ or $C=32$). Under \textbf{Protocol~C} (common), we take a further center-crop down to $32 \times 32$ for all four datasets. Because frequency features change with image size, Protocol~C gives us a fair way to compare across datasets at the same resolution.

\subsection{Spatial Stream}

The spatial stream takes the cropped RGB tensor $\mathbf{X}_{\text{crop}} \in \mathbb{R}^{B \times 3 \times H \times W}$ in the range $[0, 1]$. We normalize it inside the model using standard ImageNet mean and standard deviation:
\begin{equation}
\label{eq:imagenet_norm}
\hat{\mathbf{X}}_c = \frac{\mathbf{X}_c - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}
\end{equation}
where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

We use a ResNet-18~\cite{he2016resnet} backbone pre-trained on ImageNet-1K, with its final linear layer removed. The image passes through the initial convolution and pooling layers, followed by the four residual stages. We then apply adaptive global average pooling:
\begin{equation}
\label{eq:gap_spatial}
\mathbf{z}_s = \text{AdaptiveAvgPool2D}(\text{layer4}(\hat{\mathbf{X}})) \in \mathbb{R}^{B \times 512}
\end{equation}
Because global pooling shrinks the feature map to $1 \times 1$ regardless of input size, this branch works just as well on $224 \times 224$ crops as it does on $32 \times 32$ crops.

\subsection{Frequency Stream}

The frequency stream searches for spectral cues and periodic traces. We first turn the RGB image into a grayscale image using standard weights:
\begin{equation}
\label{eq:grayscale}
Y = 0.299\,R + 0.587\,G + 0.114\,B
\end{equation}

When we crop an image, the sharp square edges can cause frequency leaks. To smooth this out, we apply an optional 2D Hann window:
\begin{equation}
\label{eq:hann}
w(n, m) = \sin^2\!\left(\frac{\pi n}{H}\right)\,\sin^2\!\left(\frac{\pi m}{W}\right)
\end{equation}
so the input becomes $\tilde{Y} = Y \odot w$.

We test two different frequency transforms on the GPU:

\textbf{1) 2D Fast Fourier Transform (FFT):} We compute the 2D Fourier transform using \texttt{torch.fft.rfft2}. We shift the zero-frequency component to the middle using \texttt{fftshift} along the vertical axis, then pad it so the tensor matches the original $H \times W$ shape.

\textbf{2) 2D Discrete Cosine Transform (DCT-II):} For an image of size $N$, we build an orthonormal DCT-II matrix $\mathbf{D} \in \mathbb{R}^{N \times N}$:
\begin{equation}
\label{eq:dct}
D_{k,n} = s_k \cos\!\left(\frac{\pi(2n+1)k}{2N}\right), \quad s_k = \begin{cases} \sqrt{1/N} & k = 0 \\ \sqrt{2/N} & k \geq 1 \end{cases}
\end{equation}
We compute the 2D transform as two matrix multiplications:
\begin{equation}
\label{eq:separable_dct}
\mathbf{C} = \mathbf{D}\,\mathbf{X}\,\mathbf{D}^\mathsf{T}
\end{equation}
We store $\mathbf{D}$ as a fixed buffer on the GPU. We build it once for each image size and cache it. This separable design is fast and avoids the huge memory cost of building a full $N^2 \times N^2$ matrix. We tested our code against \texttt{scipy.fft.dctn} and confirmed that the outputs match closely, with differences below $5 \times 10^{-7}$.

For both transforms, we take the log magnitude and normalize the values between 0 and 1:
\begin{equation}
\label{eq:logspec}
\mathbf{S} = \frac{\log(1 + |\mathbf{F}|) - \min(\log(1 + |\mathbf{F}|))}{\max(\log(1 + |\mathbf{F}|)) - \min(\log(1 + |\mathbf{F}|)) + \epsilon}
\end{equation}
where $\mathbf{F}$ is the transform matrix and $\epsilon = 10^{-8}$.

We then feed this normalized map into a four-stage convolutional network. Each stage has a $3 \times 3$ convolution, batch normalization, and a ReLU activation. The first three stages also use $2 \times 2$ max-pooling. The channels grow from $1 \to 32 \to 64 \to 128 \to 256$. A final global average pooling layer turns this into a 256-dimensional vector $\mathbf{z}_f \in \mathbb{R}^{B \times 256}$.

\subsection{Projection and Classification}

The spatial branch outputs 512 features, while the frequency branch outputs 256 features. If we joined them directly, the spatial branch might dominate the learning process. To keep them balanced, we pass each through a small projection head:
\begin{align}
\mathbf{p}_s &= \text{ReLU}(\text{BN}(\mathbf{W}_s \mathbf{z}_s + \mathbf{b}_s)) \in \mathbb{R}^{B \times 256} \\
\mathbf{p}_f &= \text{ReLU}(\text{BN}(\mathbf{W}_f \mathbf{z}_f + \mathbf{b}_f)) \in \mathbb{R}^{B \times 256}
\end{align}
where $\mathbf{W}_s \in \mathbb{R}^{256 \times 512}$ and $\mathbf{W}_f \in \mathbb{R}^{256 \times 256}$.

We then concatenate them into a single 512-dimensional vector $\mathbf{z} = [\mathbf{p}_s \,\|\, \mathbf{p}_f]$ and pass it to our final classification head:
\begin{equation}
\label{eq:fusion}
\hat{y} = \mathbf{W}_2\,\text{Dropout}_{0.4}\!\left(\text{ReLU}(\text{BN}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1))\right) + b_2
\end{equation}
This gives us a single logit $\hat{y}$. We train the whole model using binary cross-entropy loss:
\begin{equation}
\label{eq:bce}
\mathcal{L}_{\text{BCE}}(\hat{y}, y) = \max(\hat{y}, 0) - \hat{y} y + \log(1 + e^{-|\hat{y}|})
\end{equation}
where $y = 0$ means the image is real, and $y = 1$ means it was made by AI.

% ── Algorithm ─────────────────────────────────────────────────────────────────
\begin{algorithm}[t]
\caption{Model Training and Validation Thresholding}
\label{alg:training}
\begin{algorithmic}[1]
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
\REQUIRE Training set $\mathcal{D}_{\text{train}}$, validation set $\mathcal{D}_{\text{val}}$, test set $\mathcal{D}_{\text{test}}$, seeds $\mathcal{S} = \{42, 123, 2025\}$
\ENSURE Mean and standard deviation across seeds for all metrics
\FOR{$s \in \mathcal{S}$}
    \STATE Set seed $s$ for Python, NumPy, PyTorch, and CUDA
    \STATE Build FusedDetector model with ResNet-18 and spectral CNN
    \STATE Set AdamW optimizer ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-2}$)
    \STATE Set CosineAnnealingLR scheduler ($T_{\max} = 20$, $\eta_{\min} = 10^{-6}$)
    \FOR{epoch $= 1$ \textbf{to} $E_{\max} = 20$}
        \STATE Train model with FP16 precision and clip gradients at $1.0$
        \STATE Run model on validation set to get predicted probabilities
        \STATE Calculate validation ROC-AUC
        \IF{validation ROC-AUC is the best so far}
            \STATE Save model weights; reset patience counter to 0
        \ELSE
            \STATE Add 1 to patience counter; stop training if counter reaches 5
        \ENDIF
        \STATE Step the learning rate scheduler
    \ENDFOR
    \STATE Load best model weights from checkpoint
    \STATE Find best Youden threshold $\tau^*$ on validation data:
    \STATE \quad $\tau^* = \arg\max_\tau [\text{TPR}(\tau) - \text{FPR}(\tau)]$
    \STATE Test model on $\mathcal{D}_{\text{test}}$ using the frozen threshold $\tau^*$
\ENDFOR
\STATE Calculate mean $\pm$ standard deviation over all runs
\end{algorithmic}
\end{algorithm}


% ══════════════════════════════════════════════════════════════════════════════
%  IV. EXPERIMENTAL SETUP
% ══════════════════════════════════════════════════════════════════════════════

\section{Experimental Setup}
\label{sec:setup}

\subsection{Datasets}

We run our tests across four public Kaggle benchmarks:

\begin{enumerate}[leftmargin=*,nosep]
\item \textbf{GRAVEX-200K}~\cite{wang2025sffnet}: Contains photos and modern diffusion images up to $1024 \times 1024$. We crop them to $224 \times 224$ and use this as our main training set.
\item \textbf{140K Real and Fake Faces}: Contains StyleGAN human faces paired with real FFHQ portraits at $1024 \times 1024$. We crop them to $224 \times 224$ to test how well our model transfers to faces.
\item \textbf{CASHBOWMAN}: Contains mixed Midjourney and Stable Diffusion pictures with real photos at varied sizes. We crop them to $224 \times 224$.
\item \textbf{CIFAKE}: Contains small autoencoder images and CIFAR-10 real photos at native $32 \times 32$ resolution. We crop them to $32 \times 32$.
\end{enumerate}

After removing duplicate images and balancing the classes, we have 255{,}264 images in total: 120{,}652 for training, 64{,}900 for validation, and 69{,}712 for testing. We confirmed that no image appears in more than one split.

\subsection{Training Setup}

We ran all tests on an NVIDIA Tesla T4 GPU (16~GB VRAM) on Google Colab. We train for up to 20 epochs with a batch size of 64 using PyTorch mixed precision (FP16). We use the AdamW optimizer~\cite{loshchilov2019adamw} with a learning rate of $10^{-4}$ and weight decay of $10^{-2}$. A cosine scheduler slowly lowers the learning rate down to $10^{-6}$. We stop training early if validation ROC-AUC does not improve for 5 epochs in a row. Table~\ref{tab:hyperparams} lists all key settings.

\begin{table}[t]
\caption{Hyperparameter Configuration}
\label{tab:hyperparams}
\centering
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Backbone & ResNet-18 (ImageNet pre-trained) \\
Batch size & 64 \\
Optimizer & AdamW ($\eta{=}10^{-4}$, $\lambda{=}10^{-2}$) \\
Scheduler & CosineAnnealingLR ($T_{\max}{=}20$) \\
Max epochs & 20 \\
Early stopping patience & 5 (on val ROC-AUC) \\
Gradient clipping & $\|\nabla\|_2 \leq 1.0$ \\
Mixed precision & FP16 (PyTorch AMP) \\
Projection dimension & 256 \\
Freq.\ CNN channels & 32--64--128--256 \\
Dropout (fusion head) & 0.4 \\
Hann window & Enabled (default) \\
Seeds & $\{42, 123, 2025\}$ \\
Protocol~C crop & $32 \times 32$ \\
Loss function & BCEWithLogitsLoss \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Metrics and Thresholds}

We report \textbf{ROC-AUC} and \textbf{PR-AUC} because they look at all possible thresholds and give a fair ranking.

For Accuracy and F1-score, using a fixed threshold of 0.5 is not always ideal when testing on new datasets. Instead, we use Youden's $J$ statistic~\cite{youden1950index} to find the best threshold on the validation set:
\begin{equation}
\label{eq:youden}
\tau^* = \arg\max_\tau \left[\text{TPR}(\tau) - \text{FPR}(\tau)\right]
\end{equation}
We freeze this threshold and use it as-is on the test set. We also measure how long it takes to process an image on the GPU, taking the average over 50 runs after 10 warm-up runs.


% ══════════════════════════════════════════════════════════════════════════════
%  V. RESULTS AND DISCUSSION
% ══════════════════════════════════════════════════════════════════════════════

\section{Results and Discussion}
\label{sec:results}

\subsection{In-Domain Ablation Study}

Table~\ref{tab:ablation} shows how the different model setups performed on the GRAVEX test set. The fused model with DCT scored the highest, reaching 0.8473 accuracy, 0.8508 F1-score, and 0.9268 ROC-AUC. This is slightly better than the spatial-only ResNet-18, which reached 0.9255 ROC-AUC. The frequency-only model reached 0.7021 ROC-AUC. This shows that while frequency clues are helpful, they are not strong enough on their own to tell real and fake scenes apart. Combining the two streams brings the accuracy back up and gives the model extra spectral cues.

\begin{table}[t]
\caption{In-Domain Ablation on GRAVEX Test Set (mean $\pm$ std over seeds)}
\label{tab:ablation}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Seeds} & \textbf{Accuracy} & \textbf{ROC-AUC} & \textbf{F1} \\
\midrule
Spatial-only (ResNet-18)     & 3 & 0.8451 $\pm$ 0.0012 & 0.9255 $\pm$ 0.0011 & 0.8469 $\pm$ 0.0014 \\
Freq-only (FFT+Hann)         & 3 & 0.6486 $\pm$ 0.0025 & 0.7021 $\pm$ 0.0026 & 0.6510 $\pm$ 0.0091 \\
Fused FFT+Hann               & 4 & 0.8455 $\pm$ 0.0002 & 0.9247 $\pm$ 0.0002 & 0.8469 $\pm$ 0.0006 \\
Fused FFT (no Hann)           & 1 & 0.8439              & 0.9245              & 0.8412              \\
\textbf{Fused DCT+Hann}       & 1 & \textbf{0.8473}     & \textbf{0.9268}     & \textbf{0.8508}     \\
\bottomrule
\end{tabular}%
}
\end{table}

We also tested the model with and without the Hann window. The difference was very small: 0.9247 with the window and 0.9245 without it. This tells us that boundary leaks during cropping have only a minor effect compared to the main generator artifacts. The results were very consistent across seeds, with standard deviations below 0.0012 for the fused runs.

\subsection{Speed and Model Size}

Table~\ref{tab:efficiency} shows the parameter counts and latency numbers on our Tesla T4 GPU. Adding the frequency stream adds very little cost. The model size only goes up by 0.10M parameters, from 11.86M for the spatial model to 11.96M for the fused model. This is because our frequency CNN uses relatively small feature maps.

\begin{table}[t]
\caption{Computational Efficiency Comparison}
\label{tab:efficiency}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{Params (M)} & \textbf{Lat.\ BS=1 (ms)} & \textbf{Lat.\ BS=64 (ms/img)} \\
\midrule
Freq-only (FFT)        & 11.86 & 2.76  & 0.046 \\
Spatial-only            & 11.86 & 5.99  & 0.085 \\
Fused FFT (no Hann)     & 11.96 & 7.70  & 0.138 \\
Fused FFT+Hann          & 11.96 & 7.68  & 0.132 \\
Fused DCT+Hann          & 11.96 & 9.13  & 0.154 \\
\bottomrule
\end{tabular}%
}
\end{table}

When processing batches of 64 images, the fused DCT model takes only 0.154~ms per image, which means it can process more than 6{,}490 images per second. The FFT model is slightly faster at 0.132~ms per image. For a single image, the DCT model takes 9.13~ms, while the FFT model takes 7.68~ms. Both run fast enough for real-time video, which usually allows up to 33.3~ms per frame. The DCT model is slightly slower because of the matrix multiplication on the GPU, whereas the FFT takes advantage of built-in cuFFT libraries.

\subsection{Testing on New Datasets}

Table~\ref{tab:cross} shows how our GRAVEX-trained model performed on the other three datasets under Protocol~C ($32 \times 32$ crops). We used the threshold from the GRAVEX validation set without any tuning.

\begin{table}[t]
\caption{Zero-Shot Cross-Dataset Transfer (Protocol~C, $32{\times}32$)}
\label{tab:cross}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Target} & \textbf{Setting} & \textbf{Samples} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{Acc.\ (\%)} \\
\midrule
GRAVEX      & In-Domain   & 29{,}991 & 0.9421 & 0.9433 & 87.03 \\
FACES       & Zero-Shot   & 20{,}000 & \textbf{0.9656} & \textbf{0.9683} & \textbf{90.97} \\
CASHBOWMAN  & Zero-Shot   & 137      & 0.4603 & 0.5773 & 54.01 \\
CIFAKE      & Zero-Shot   & 19{,}622 & 0.5001 & 0.4944 & 49.44 \\
\bottomrule
\end{tabular}%
}
\end{table}

On the 140K Faces dataset, the model transferred very well. It reached \textbf{0.9656 ROC-AUC} and \textbf{90.97\% accuracy}. This is an encouraging result. The model was trained on general diffusion images (like landscapes, animals, and objects) and had never seen any of these human faces during training. Yet it detected the StyleGAN faces with high accuracy. This suggests that some upsampling artifacts and frequency traces are shared across different generative pipelines.

However, the model struggled on CASHBOWMAN (0.4603 AUC) and CIFAKE (0.5001 AUC). For CASHBOWMAN, the test set has only 137 images, so the sample size is too small to draw firm conclusions. But for CIFAKE, the issue is clear: it comes down to image resolution. GRAVEX images are large, with over 50{,}000 pixels in a crop. CIFAKE images are tiny, with only $32 \times 32$ pixels (1{,}024 pixels in total). When we crop them down to $32 \times 32$, the model sees tiny pixelated objects instead of the rich textures it saw during training. This shows that differences in image resolution between datasets can cause detectors to fail, even when the model architecture itself is sound.

\subsection{Limitations}

We should point out a few limits of our work. First, we trained our model on just one main dataset. Training on multiple datasets at once would likely make the model more robust. Second, center-cropping avoids resizing artifacts, but it throws away the outer parts of the image and covers a different portion of the scene when images have different original sizes. Third, while our SHA-256 check catches exact duplicate files, it does not flag images that look almost identical but differ by minor pixel noise. Finally, our images come from Kaggle, so they may not reflect all the re-compression and downscaling that happens when people share pictures on social media platforms.


% ══════════════════════════════════════════════════════════════════════════════
%  VI. CONCLUSION
% ══════════════════════════════════════════════════════════════════════════════

\section{Conclusion and Future Work}
\label{sec:conclusion}

In this paper, we explored whether combining spatial and frequency features can improve AI-generated image detection. We built a dual-branch network that pairs a pre-trained ResNet-18 with a small spectral CNN using 2D-FFT or separable 2D-DCT transforms. We used center-crops without resizing and checked file hashes to prevent any data leakage. Our model reached 0.9268 ROC-AUC on the GRAVEX test set and transferred well to unseen StyleGAN faces, achieving 0.9656 ROC-AUC and 90.97\% accuracy. At the same time, our tests showed that models trained on high-resolution data struggle when tested on small images like CIFAKE.

In future work, we plan to test three main ideas: (1) adding random JPEG and WebP compression during training to help the model handle online image sharing, (2) using multi-scale patch attention to better handle images of different resolutions without resizing, and (3) training across several different generative models at once to build a more universal detector.


% ══════════════════════════════════════════════════════════════════════════════
%  REFERENCES
% ══════════════════════════════════════════════════════════════════════════════

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

with open("paper/main.tex", "w", encoding="utf-8") as f:
    f.write(natural_tex)

print("Saved natural humanized main.tex!")
analyze_prose(natural_tex)
