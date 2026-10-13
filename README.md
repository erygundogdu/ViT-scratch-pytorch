# Vision Transformer (ViT)

A PyTorch image classifier implemented in [`vit_arch.py`](vit_arch.py). Images become patch tokens, pass through Transformer encoder blocks, and are classified using their final CLS tokens.

## Architecture block diagram

```mermaid
flowchart TD
    A["Image batch: B x 3 x 640 x 640"] --> B["Split each image into 1,600 patches<br/>Patch size: 16 x 16"]
    B --> C["Flatten patches: B x 1,600 x 768"]
    C --> D["Linear projection: 768 to 512"]
    D --> E["Prepend learnable CLS token<br/>Add positional embeddings<br/>B x 1,601 x 512"]
    E --> G
    subgraph Encoder["Transformer encoder block repeated 4 times"]
        G["Block input"] --> H[LayerNorm]
        H --> I["Multi-head self-attention<br/>8 heads, 64 dimensions per head"]
        I --> J[Residual addition]
        G --> J
        J --> K[LayerNorm]
        K --> L["MLP: 512 to 2,048 to 512<br/>GELU activation"]
        L --> M[Residual addition]
        J --> M
    end
    M --> N["Select CLS token per image: B x 512"]
    N --> O[LayerNorm]
    O --> P["Classification MLP: 512 to 1,024 to 3<br/>GELU activation"]
    P --> Q["Class logits: B x 3"]
```

## Implementation summary

- **Embedding:** projects non-overlapping patches into 512 dimensions and adds a learnable CLS token and positional embeddings.
- **Encoder:** uses pre-normalization, scaled dot-product attention, and an MLP, with residual connections around attention and the MLP.
- **Head:** normalizes each final CLS token and produces three class logits without softmax.

The example uses `l=4`, `d=512`, `num_heads=8`, and `cls=3`. Input shape is `(B, 3, 640, 640)`; output shape is `(B, 3)`. The included batch of four images produces `(4, 3)`. Image size is fixed to 640 x 640, and `d` must be divisible by `num_heads`.
