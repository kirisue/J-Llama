package com.kirisu.llama2;

import java.nio.FloatBuffer;

/**
 * Neural network weights (1D arrays for cache locality).
 */
public class TransformerWeights {
    public float[] token_embedding_table;
    public float[] rms_att_weight;
    public float[] rms_ffn_weight;

    public float[] wq;
    public float[] wk;
    public float[] wv;
    public float[] wo;

    public float[] w1;
    public float[] w2;
    public float[] w3;

    public float[] rms_final_weight;
    public float[] wcls;

    public static TransformerWeights load(Config config, FloatBuffer buffer) {
        TransformerWeights w = new TransformerWeights();
        int head_size = config.dim / config.n_heads;

        w.token_embedding_table = readFloatArray(buffer, config.vocab_size * config.dim);
        w.rms_att_weight = readFloatArray(buffer, config.n_layers * config.dim);
        w.wq = readFloatArray(buffer, config.n_layers * config.dim * (config.n_heads * head_size));
        w.wk = readFloatArray(buffer, config.n_layers * config.dim * (config.n_kv_heads * head_size));
        w.wv = readFloatArray(buffer, config.n_layers * config.dim * (config.n_kv_heads * head_size));
        w.wo = readFloatArray(buffer, config.n_layers * (config.n_heads * head_size) * config.dim);
        w.rms_ffn_weight = readFloatArray(buffer, config.n_layers * config.dim);
        w.w1 = readFloatArray(buffer, config.n_layers * config.dim * config.hidden_dim);
        w.w2 = readFloatArray(buffer, config.n_layers * config.hidden_dim * config.dim);
        w.w3 = readFloatArray(buffer, config.n_layers * config.dim * config.hidden_dim);
        w.rms_final_weight = readFloatArray(buffer, config.dim);

        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        buffer.position(buffer.position() + (config.seq_len * head_size / 2));

        if (shared_weights == 1) {
            w.wcls = w.token_embedding_table; // Weight tying
        } else {
            w.wcls = readFloatArray(buffer, Math.abs(config.vocab_size) * config.dim);
        }

        return w;
    }

    private static float[] readFloatArray(FloatBuffer buffer, int size) {
        float[] arr = new float[size];
        buffer.get(arr);
        return arr;
    }
}