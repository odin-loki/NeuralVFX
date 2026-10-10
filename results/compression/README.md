# Lossless packing of effect files: context mixing, and flipbooks coded the same way

Status: **results** (9 October 2026). Audience: owner, research, dev. Data: [cm.csv](cm.csv) (every file, flipbook and
part) and [cm_equal_quality.csv](cm_equal_quality.csv). Code: `include/neuralfx/cm.hpp`, `src/core/cm.cpp`,
`tools/nvfx_pack.cpp`.

## Bottom line

- **The network files shrink by 1.2 to 1.7 times (frame models) and 1.8 to 4.9 times (rollout effects), losslessly.**
  zlib -9 manages 1.07 to 1.23 times on the frame models and 1.3 to 2.6 times on the rollout effects.
- **Flipbooks shrink far more: 2.9 to 6.9 times in BC3 layout and 6.9 to 12.9 times as raw RGBA** (zlib -9: 2.5 to
  4.9 times). A sprite flipbook is mostly empty, smooth and repeated in time; a trained network's 8-bit features are
  close to noise (7.1 to 7.3 bits per byte at order 0).
- **So entropy coding does not push the network's advantage past 10 times; it erodes it.** At equal quality (mean active
  PSNR, the comparison of [docs/REPORT.md](../../docs/REPORT.md) §3), with both sides coded by this coder, the 8-bit
  networks need **1.4 times less disk than a flipbook at best** (grid_m 132 KB and conv_s 69 KB), tie at grid_s
  (1.0x), and need *more* disk for conv_m (0.9x) and grid_mt (0.8x). In memory the same networks need 3.5 to 5.9
  times less. fp16 networks lose on disk (0.4 to 0.8x).
- **Resident memory does not change.** The runtime unpacks at load to the same `.nvfx` bytes, so RAM and the
  per-frame cost are exactly as in the report. Only disk and download sizes change.
- **Decoding is slow: 0.32 to 0.44 MB/s on one core** (about 1,700 instructions per coded bit, typical of this class of
  coder). A 132 KB study A model takes 0.4 s to unpack, a 1 MB study B model 3.2 s.
- **To gain from entropy coding, the networks would have to be trained for it**: a rate term in the training loss (as
  learned image and texture codecs use) would make the features predictable instead of noise-like. That is a change to
  training, not to the coder; this coder would then measure it.
- **A correction to the report**: the equal-quality ratio of conv_s is **4.9x, not 7.4x**. The report's envelope
  interpolated between two flipbooks of the same size (512 KB); see "Equal quality" below. The other ratios reproduce.

## The coder

A PAQ/lpaq-style context-mixing coder that knows the tensors' shapes. Every value is coded bit by bit, most significant
first, with a 32-bit carry-less binary arithmetic coder (16-bit probabilities). Each bit's probability comes from:

- **Numeric prediction per value.** 13 fixed predictors (left, above, gradient, median edge detector, the previous plane
  at the same place, one row down and with the local gradient, the plane two axes back, and for pixels and blocks the
  previous channel's change) plus an adaptive linear predictor (normalised LMS on 12 neighbours), blended by the
  inverse square of each predictor's recent error at the four causal neighbours and in the previous plane. fp16 values
  are predicted in exact fixed point (value x 2^24); an 8-bit neighbour in another plane is first mapped into this
  plane's scale through both planes' stored (lo, hi) ranges.
- **3 direct context models per bit:** where the prediction lies relative to the middle of the interval still open
  (in eighths of its half-width) with the local error level; the same offset in units of the local error; and the
  offset of the locally best single predictor.
- **14 hashed context models per byte** (slots of one cache line per nibble, two-way, checked): order 0; the value to
  the left, above, in the previous plane; left and above together; the quantised prediction, with and without the
  error level; left, above and above-right coarsely; per tensor and plane, per row, per column (weights of one neuron
  or one input share a scale); the bytes of this pixel or block decoded so far; the previous plane with its row below.
  BC3 blocks swap five of these for block-aware ones: the same byte in the blocks to the left, above and in the
  previous frame; whether the alpha or colour endpoints are equal or zero; the block's bytes so far with its left and
  previous-frame neighbours.
- **Two mixers** (logistic, 18 inputs, 16-bit weights with rounded updates), one selected by kind, bit position and
  error level, one by kind, byte and the bits decoded so far, averaged in the logistic domain.
- **Two APMs** (adaptive probability maps): by the bits decoded so far, and by the prediction's offset.
- **fp16** is reordered by a bijection that is monotone in the value, then coded as the high byte (sign, exponent and
  two mantissa bits) and the low byte, with the high byte in every context of the low byte.
- **The container** parses a `.nvfx` (either kind) into its tensors and codes each with its shape: features in the order
  [channel][basis][time][y][x] (so "the previous plane" is the previous time slice and two axes back the previous
  basis), layers as [out][in], start states as [start][y][x][channel], fine fields with their scales. Headers are coded
  as plain bytes. A 64-bit checksum of the original guards the round trip; anything that does not parse as a model is
  coded as plain bytes.

Everything the decoder computes is integer arithmetic in a fixed order, so every compiler and instruction set decodes
the same bits. Checked: the coder built by GCC 14 at -O3 for this machine's AVX-512, by Clang 18 at -O2 for baseline
x86-64, and by GCC with the address and undefined-behaviour sanitisers at -O1 writes byte-identical streams for the
same files. The tests (`tests/test_cm.cpp`) check exact round trips on random and smooth tensors of many shapes,
frame models and rollout effects saved in memory, empty, tiny, truncated and malformed inputs, and that damaged data
is refused rather than decoded into a different file (a damaged stream is detected by the checksum, or earlier when
the decoder runs past the end of its input). A sanitiser run over 180 damaged model streams, 200 random tensor sets
and 4,000 junk streams found no fault.

## Models

Every model under the data root (A: one model per clip, every configuration for clip 0 of each effect and grid_m 8-bit
for all 12 clips; B: control models; C: variation models; D: rollout effects). Means over the files of each
configuration; bits per value counts every stored value (features, weights, states), headers excluded.

| study | config | files | KB | packed KB | ratio | bits per value (stored / packed) | zlib -9 ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| A | grid_s 8-bit | 3 | 73.1 | 48.0 | 1.52x | 8.07 / 5.30 | 1.11x |
| A | grid_m 8-bit | 12 | 131.7 | 85.0 | 1.55x | 8.12 / 5.24 | 1.17x |
| A | grid_mt 8-bit | 3 | 260.2 | 180.1 | 1.44x | 8.07 / 5.59 | 1.13x |
| A | grid_l 8-bit | 3 | 291.7 | 175.0 | 1.67x | 8.05 / 4.83 | 1.23x |
| A | conv_s 8-bit | 3 | 69.2 | 57.2 | 1.21x | 8.32 / 6.88 | 1.07x |
| A | conv_m 8-bit | 3 | 142.1 | 117.6 | 1.21x | 8.42 / 6.97 | 1.08x |
| A | grid_s fp16 | 3 | 144.6 | 110.9 | 1.30x | 16.02 / 12.28 | 1.08x |
| A | grid_m fp16 | 3 | 259.2 | 195.7 | 1.32x | 16.01 / 12.09 | 1.09x |
| A | grid_mt fp16 | 3 | 515.2 | 385.1 | 1.34x | 16.01 / 11.96 | 1.11x |
| A | grid_l fp16 | 3 | 579.2 | 430.5 | 1.35x | 16.00 / 11.90 | 1.10x |
| A | conv_s fp16 | 3 | 132.2 | 108.0 | 1.22x | 16.02 / 13.09 | 1.08x |
| A | conv_m fp16 | 3 | 268.1 | 220.6 | 1.22x | 16.01 / 13.17 | 1.08x |
| B | grid k8 (1 MB) | 3 | 1031.7 | 794.6 | 1.30x | 8.03 / 6.19 | 1.09x |
| B | grid k16 | 3 | 2062.9 | 1562.8 | 1.32x | 8.03 / 6.08 | 1.09x |
| C | variation k8 | 3 | 1032.7 | 725.9 | 1.42x | 8.03 / 5.65 | 1.12x |
| C | variation k24 | 3 | 3089.0 | 2165.3 | 1.43x | 8.02 / 5.62 | 1.13x |
| D | rollout fire | 1 | 81.8 | 44.5 | 1.84x | 16.08 / 8.76 | 1.29x |
| D | rollout smoke | 1 | 145.8 | 50.3 | 2.90x | 11.14 / 3.84 | 2.01x |
| D | rollout explosion | 1 | 274.3 | 55.7 | 4.92x | 10.92 / 2.22 | 2.64x |

By part (means per file; packed sizes are the model's code lengths, the arithmetic coder adds a few bytes):

| model | part | KB | packed KB | ratio | zlib -9 ratio |
|---|---|---:|---:|---:|---:|
| A grid_m 8-bit | features | 128.0 | 82.1 | 1.56x | 1.20x |
| A grid_m 8-bit | feature ranges | 0.5 | 0.4 | 1.42x | 1.02x |
| A grid_m 8-bit | MLP weights | 2.8 | 2.3 | 1.18x | 1.05x |
| A grid_m fp16 | features | 256.0 | 193.2 | 1.33x | 1.12x |
| A conv_s 8-bit | features | 64.0 | 53.0 | 1.21x | 1.08x |
| A conv_s 8-bit | convolution weights | 3.9 | 3.3 | 1.18x | 1.06x |
| B grid k8 | features | 1024.0 | 789.3 | 1.30x | 1.10x |
| C variation k24 | features | 3072.0 | 2154.4 | 1.43x | 1.15x |
| D rollout (3) | stepper and renderer weights | 17.3 | 14.8 | 1.17x | 1.07x |
| D rollout (3) | coarse start states (fp16) | 85.3 | 31.4 | 2.72x | 1.49x |
| D rollout (2 with them) | fine start fields (8-bit) | 96.0 | 5.4 | 17.90x | 12.72x |

What compresses and what does not:
- **8-bit grid features: 4.8 to 5.5 bits per value in study A, 5.6 in C, 6.1 to 6.2 in B.** They are near noise at
  order 0 (7.1 to 7.3 bits); the gain comes from spatial prediction and from the previous time slice and basis once
  mapped into the same scale. Conv latents barely compress (6.6 bits): they have almost no spatial correlation.
- **fp16 weights: 13.7 bits per value.** The high byte (sign and exponent) carries the gain; the low mantissa byte is
  noise. The same holds for fp16 features (11.9 to 13.2 bits): the extra 8 bits over the 8-bit version are noise.
- **Rollout start points compress well**: coarse states are smooth fields (4.5 to 7.4 bits per fp16 value) and the
  8-bit fine fields are mostly empty (0.3 to 0.8 bits per value). So the more of an effect is start points, the more it
  shrinks: explosion 4.9x, smoke 2.9x, fire (no fine fields) 1.8x.

## Flipbooks, coded the same way

The study A flipbook ladder for the 12 study A clips, encoded by `src/core/flipbook.cpp` exactly as the study stored
them: BC3-layout blocks as a tensor [frame][block row][block column][16 bytes], raw RGBA as [frame][y][x][4], motion
vectors as [frame][y][x][2] (quarter pixels, 8 bits). Means over the 12 clips; the full ladder (31 configurations) is in
cm.csv.

| flipbook | KB in memory | packed KB | ratio | zlib -9 KB | zlib ratio |
|---|---:|---:|---:|---:|---:|
| BC3 64 frames 128 px (every frame) | 1024 | 148.2 | 6.91x | 265.8 | 3.85x |
| BC3 32 frames 128 px | 512 | 85.3 | 6.00x | 136.3 | 3.76x |
| BC3 16 frames 128 px + motion vectors | 288 | 51.6 | 5.58x | 79.3 | 3.63x |
| BC3 16 frames 128 px | 256 | 46.8 | 5.47x | 68.9 | 3.71x |
| BC3 64 frames 64 px | 256 | 44.5 | 5.75x | 74.8 | 3.42x |
| BC3 32 frames 64 px | 128 | 26.8 | 4.78x | 39.5 | 3.24x |
| BC3 16 frames 64 px + motion vectors | 72 | 16.6 | 4.34x | 24.1 | 2.99x |
| BC3 64 frames 32 px | 64 | 12.9 | 4.97x | 21.4 | 2.99x |
| raw RGBA 64 frames 64 px | 1024 | 79.4 | 12.89x | 213.7 | 4.79x |
| raw RGBA 32 frames 64 px | 512 | 47.0 | 10.88x | 108.3 | 4.73x |
| raw RGBA 64 frames 32 px | 256 | 25.3 | 10.10x | 60.5 | 4.23x |

By effect, the 1 MB BC3 flipbook packs 8.67x (fire), 5.45x (smoke) and 7.39x (explosion); at 64 px 6.89x, 4.77x and
6.00x; 32 frames at 128 px 7.71x, 4.63x and 6.50x. The BC3 model is generic (bytes with block-aware contexts); a coder
that decoded the blocks to predict endpoints from neighbouring pixels, or an encoder tuned for rate (as production
texture pipelines use), would shrink flipbooks further, so these flipbook sizes are an upper bound.

## Equal quality: memory against disk

For each study A network, the flipbook size that reaches the same mean active PSNR over the 12 clips, along the
best-flipbook envelope (the best quality at or below each size, log-linear between sizes), as in the report; then the
same with every flipbook and every network packed by this coder, and with zlib -9. Network sizes are the means of the
saved files (grid_m 8-bit: 12 clips; the others: clip 0 of each effect). Ratio = flipbook size / network size.

| network | active PSNR | in memory: network / flipbook KB, ratio | packed: network / flipbook KB, ratio | zlib -9: network / flipbook KB, ratio |
|---|---:|---|---|---|
| conv_s 8-bit | 29.45 | 69 / 341, **4.9x** | 57 / 81, **1.4x** | 64 / 116, 1.8x |
| grid_s 8-bit | 28.07 | 73 / 265, **3.6x** | 48 / 48, **1.0x** | 66 / 76, 1.2x |
| grid_m 8-bit | 32.62 | 132 / 772, **5.9x** | 85 / 118, **1.4x** | 113 / 243, 2.2x |
| conv_m 8-bit | 31.33 | 142 / 634, **4.5x** | 118 / 101, **0.9x** | 132 / 229, 1.7x |
| grid_mt 8-bit | 33.76 | 260 / 917, **3.5x** | 180 / 136, **0.8x** | 231 / 257, 1.1x |
| grid_l 8-bit | 36.23 | 292 / > 1024, **> 3.5x** | 175 / > 148, **> 0.8x** | 236 / > 266, > 1.1x |
| conv_s fp16 | 29.45 | 132 / 343, 2.6x | 108 / 81, 0.8x | 122 / 116, 1.0x |
| grid_s fp16 | 28.07 | 145 / 265, 1.8x | 111 / 48, 0.4x | 133 / 76, 0.6x |
| grid_m fp16 | 32.64 | 259 / 774, 3.0x | 196 / 119, 0.6x | 238 / 243, 1.0x |
| conv_m fp16 | 31.33 | 268 / 634, 2.4x | 221 / 101, 0.5x | 248 / 229, 0.9x |
| grid_mt fp16 | 33.77 | 515 / 920, 1.8x | 385 / 136, 0.4x | 465 / 257, 0.6x |
| grid_l fp16 | 36.28 | 579 / > 1024, > 1.8x | 431 / > 148, > 0.3x | 524 / > 266, > 0.5x |

(">": no flipbook in the ladder is as good, so the flipbook size and the ratio are at least that. grid_l is above the
1 MB BC3 flipbook in quality, so its ratios are lower bounds.)

- **On disk, the networks' advantage is 1.4x at best, and gone for the larger ones.** Packing gives the flipbook
  ladder 2.9 to 12.9 times and the networks 1.2 to 1.7 times, so the memory ratios shrink by a factor of 3.5 to 5.
- **By effect** (grid_m 8-bit against the 1 MB BC3 flipbook): on fire the network is better (31.47 against 30.44 dB) and
  smaller on disk too (81 KB against 118 KB packed); on smoke (31.86 against 36.48 dB, 86 against 188 KB) and
  explosions (34.53 against 36.51 dB, 88 against 139 KB) the flipbook is better and is only 1.6 to 2.2 times larger.
- **The rollout effects of study D** pack to 45 to 56 KB each, a third of one packed 1 MB flipbook of one clip, and they
  play endlessly at any setting; but they are not compared at equal quality here (they do not reproduce a clip).
- **The correction.** The report's 7.4x for conv_s came from two flipbooks of the same size: raw 32 frames at 64 px
  (26.94 dB) and BC3 32 frames at 128 px (29.92 dB), both 512 KB. The report's envelope kept both points and
  interpolated between them, which lands on 512 KB for any quality in between. With one point per size (the best), the
  envelope goes from 288 KB (29.25 dB) to 512 KB (29.92 dB) and conv_s's 29.45 dB needs 341 KB: 4.9x. The tool here
  merges equal sizes, and so does `nvfx_experiment report` now; the report is corrected.

## Speed

Single thread on one pinned core of the study machine (Intel Xeon, 2.8 GHz), quiet, best of three; the coder is plain
C++ built for baseline x86-64 (no instruction-set-specific code):

| file | KB | encode MB/s | decode MB/s |
|---|---:|---:|---:|
| A conv_s 8-bit | 69 | 0.38 | 0.32 |
| A grid_m 8-bit | 132 | 0.31 | 0.35 |
| A grid_m fp16 | 259 | 0.34 | 0.34 |
| D fire | 82 | 0.39 | 0.40 |
| D explosion | 274 | 0.44 | 0.44 |
| B fire grid k8 | 1032 | 0.32 | 0.33 |

Decoding is as costly as encoding (the same model runs). It needs up to about 70 MB of working memory while it runs
(a 64 MB statistics table for inputs of 512 KB and more, less for smaller ones), released when done. The figures in
cm.csv were measured while other jobs shared the machine and spread wider (0.17 to 0.58 MB/s).

## What did not help (measured on a set of five model files and on flipbooks)

- **More of the same contexts.** Each hashed context alone is worth at most 0.3%; all 14 together 2% over order 0 plus
  the direct contexts. Most of the gain is in the direct contexts relative to the numeric prediction.
- **A lean model** (8 hashed contexts, one mixer): about 20% faster on a flipbook, 0.6% (models) to 0.8% (flipbooks)
  larger.
- **The adaptive linear predictor (NLMS):** 0.06%. Using it instead of the best single predictor for the third direct
  context: better on features, worse on fp16, net loss.
- **Fractional fp16 predictions** (where the prediction falls between two fp16 values): no gain; the low mantissa bits
  are noise either way.
- **Per-tensor statistics and shared mixers for small tensors:** 0.01% each (kept, they cost nothing).
- **Bigger statistics tables:** none for model files; 0.9% on a 1 MB flipbook going from 64 MB to 256 MB (not used).
- **Mixer settings:** higher learning rates (10, 16) were worse, a four times larger initial weight 1% worse. 16-bit
  weights with truncated updates cost 0.1% on models and 1.1% on BC3 flipbooks, with 13 fractional bits 0.4% and
  3.4%; rounding the updates removed the loss (and beat 32-bit weights slightly).

## Reproduce

```sh
build/nvfx_pack --study --data $NEURALVFX_DATA/experiments   # about 25 minutes, one core: cm.csv, cm_equal_quality.csv, tables
build/nvfx_pack --tables                                     # the tables again from cm.csv
build/nvfx_pack --report $NEURALVFX_DATA/experiments/models  # any folder of .nvfx files
build/nvfx_pack in.nvfx out.nvfz && build/nvfx_pack --unpack out.nvfz back.nvfx
```
