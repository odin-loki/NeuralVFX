# Study hres fire — exported 3D rollout models (32³ coarse)

From `$NEURALVFX_DATA/hres_3d/fire` after `hres_fire.sh` / training to iteration 1750.

| File | SHA-256 |
|---|---|
| `C1.nvfx3` | `754bbba24fedd48a22f98c4ea2c6bbb3dae6f59aeb05ad515b6041856888d18c` |
| `C2.nvfx3` | `56109791b2a686f687163acc5f89d8de6fb519667054efebcc4754653e936349` |
| `C3.nvfx3` | `29e52ad7e81823621b7c3404422eb3f861a66f7c1c943748adceefd3009ac14a` |
| `C4.nvfx3` | `4a87cb12e95c44c53c9af5b1e5896237e05ee598866f229bb4637da6dc9cbc70` |
| `hres_fire.nvfx3` | `4a87cb12e95c44c53c9af5b1e5896237e05ee598866f229bb4637da6dc9cbc70` (same weights as `C4`) |

Restore for tools: `cp results/3d/hres/fire/models/*.nvfx3 "$NEURALVFX_DATA/hres_3d/fire/models/"`
