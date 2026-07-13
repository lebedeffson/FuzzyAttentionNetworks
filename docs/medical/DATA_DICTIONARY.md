# Data Dictionary

- `model_input`: `[36, D]` model window.
- `h_ffn`: transformer FFN input activations `[L, N, 36, d_model]`.
- `a_ffn`: transformer FFN output activations `[L, N, 36, d_model]`.
- `x_observed`: first eight normalized value channels used for temporal SCTC weights.
- `split`: `0=train`, `1=validation`, `2=test`.
- `feature_catalog`: eligible sparse features and decoder statistics.
- `edge_catalog`: directed edge candidates and intervention statistics.
