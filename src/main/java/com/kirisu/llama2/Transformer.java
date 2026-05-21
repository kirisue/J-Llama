package com.kirisu.llama2;

/**
 * Transformer neural network architecture.
 */
public class Transformer {

    public static void forward(Config config, TransformerWeights w, RunState s, int token, int pos) {
        int dim = config.dim;
        int hidden_dim = config.hidden_dim;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        int kv_mul = config.n_heads / config.n_kv_heads;
        int head_size = dim / config.n_heads;

        // Token Embedding
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        for (int l = 0; l < config.n_layers; l++) {

            // Attention RMSNorm
            float[] current_rms_att = new float[dim];
            System.arraycopy(w.rms_att_weight, l * dim, current_rms_att, 0, dim);
            MathUtils.rmsnorm(s.xb, s.x, current_rms_att, dim);

            // QKV MatMuls
            float[] wq_layer = extractLayer(w.wq, l, dim * dim);
            float[] wk_layer = extractLayer(w.wk, l, dim * kv_dim);
            float[] wv_layer = extractLayer(w.wv, l, dim * kv_dim);

            MathUtils.matmul(s.q, s.xb, wq_layer, dim, dim);
            MathUtils.matmul(s.k, s.xb, wk_layer, dim, kv_dim);
            MathUtils.matmul(s.v, s.xb, wv_layer, dim, kv_dim);

            // RoPE (Rotary Positional Embedding)
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / (float)Math.pow(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = (float)Math.cos(val);
                float fci = (float)Math.sin(val);

                float q0 = s.q[i], q1 = s.q[i+1];
                s.q[i]   = q0 * fcr - q1 * fci;
                s.q[i+1] = q0 * fci + q1 * fcr;

                if (i < kv_dim) {
                    float k0 = s.k[i], k1 = s.k[i+1];
                    s.k[i]   = k0 * fcr - k1 * fci;
                    s.k[i+1] = k0 * fci + k1 * fcr;
                }
            }

            // KV Cache
            int loff = l * config.seq_len * kv_dim;
            System.arraycopy(s.k, 0, s.key_cache, loff + pos * kv_dim, kv_dim);
            System.arraycopy(s.v, 0, s.value_cache, loff + pos * kv_dim, kv_dim);

            // Multi-Head Attention
            for (int h = 0; h < config.n_heads; h++) {
                int q_offset = h * head_size;
                for (int t = 0; t <= pos; t++) {
                    int k_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[q_offset + i] * s.key_cache[k_offset + i];
                    }
                    score /= Math.sqrt(head_size);
                    s.att[h * config.seq_len + t] = score;
                }

                float[] head_att = extractLayer(s.att, h, config.seq_len);
                MathUtils.softmax(head_att, pos + 1);
                System.arraycopy(head_att, 0, s.att, h * config.seq_len, pos + 1);

                for (int i = 0; i < head_size; i++) {
                    float val = 0.0f;
                    for (int t = 0; t <= pos; t++) {
                        int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                        val += s.att[h * config.seq_len + t] * s.value_cache[v_offset + i];
                    }
                    s.xb[q_offset + i] = val;
                }
            }

            // Output projection & residual
            float[] wo_layer = extractLayer(w.wo, l, dim * dim);
            MathUtils.matmul(s.xb2, s.xb, wo_layer, dim, dim);
            for (int i = 0; i < dim; i++) {
                s.x[i] += s.xb2[i];
            }

            // FFN & SwiGLU
            float[] current_rms_ffn = new float[dim];
            System.arraycopy(w.rms_ffn_weight, l * dim, current_rms_ffn, 0, dim);
            MathUtils.rmsnorm(s.xb, s.x, current_rms_ffn, dim);

            float[] w1_layer = extractLayer(w.w1, l, dim * hidden_dim);
            float[] w3_layer = extractLayer(w.w3, l, dim * hidden_dim);
            MathUtils.matmul(s.hb, s.xb, w1_layer, dim, hidden_dim);
            MathUtils.matmul(s.hb2, s.xb, w3_layer, dim, hidden_dim);

            MathUtils.silu(s.hb, hidden_dim);
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            float[] w2_layer = extractLayer(w.w2, l, dim * hidden_dim);
            MathUtils.matmul(s.xb, s.hb, w2_layer, hidden_dim, dim);
            for (int i = 0; i < dim; i++) {
                s.x[i] += s.xb[i];
            }
        }

        // Final RMSNorm and Classifier
        MathUtils.rmsnorm(s.x, s.x, w.rms_final_weight, dim);
        MathUtils.matmul(s.logits, s.x, w.wcls, dim, config.vocab_size);
    }

    private static float[] extractLayer(float[] source, int layer, int size) {
        float[] result = new float[size];
        System.arraycopy(source, layer * size, result, 0, size);
        return result;
    }
}