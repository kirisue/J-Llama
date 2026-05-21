package com.kirisu.llama2;

/**
 * Model configuration for Llama 2 architecture.
 */
public class Config {
    public final int dim;        // Transformer dimension
    public final int hidden_dim; // FFN hidden dimension
    public final int n_layers;   // Number of layers
    public final int n_heads;    // Number of query heads
    public final int n_kv_heads; // Number of key/value heads
    public final int vocab_size; // Vocabulary size
    public final int seq_len;    // Max sequence length

    public Config(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len) {
        this.dim = dim;
        this.hidden_dim = hidden_dim;
        this.n_layers = n_layers;
        this.n_heads = n_heads;
        this.n_kv_heads = n_kv_heads;
        this.vocab_size = vocab_size;
        this.seq_len = seq_len;
    }

    @Override
    public String toString() {
        return String.format("Config [dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d]",
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len);
    }
}