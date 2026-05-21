package com.kirisu.llama2;

/**
 * Inference memory pool to prevent GC overhead during generation.
 */
public class RunState {
    public float[] x;      // activation at current layer (dim)
    public float[] xb;     // buffer inside a layer (dim)
    public float[] xb2;    // buffer inside a layer (dim)
    public float[] hb;     // buffer for hidden dimension in FFN (hidden_dim)
    public float[] hb2;    // buffer for hidden dimension in FFN (hidden_dim)
    public float[] q;      // query (dim)
    public float[] k;      // key (kv_dim)
    public float[] v;      // value (kv_dim)
    public float[] att;    // attention scores (n_heads * seq_len)
    public float[] logits; // output probabilities (vocab_size)

    // KV Cache
    public float[] key_cache;   // (layer, seq_len, kv_dim)
    public float[] value_cache; // (layer, seq_len, kv_dim)

    public RunState(Config config) {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        this.x = new float[config.dim];
        this.xb = new float[config.dim];
        this.xb2 = new float[config.dim];
        this.hb = new float[config.hidden_dim];
        this.hb2 = new float[config.hidden_dim];
        this.q = new float[config.dim];
        this.k = new float[kv_dim];
        this.v = new float[kv_dim];
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new float[config.vocab_size];

        this.key_cache = new float[config.n_layers * config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers * config.seq_len * kv_dim];
    }
}