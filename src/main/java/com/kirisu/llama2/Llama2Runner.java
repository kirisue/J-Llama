package com.kirisu.llama2;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Main entry point for the Inference Engine.
 */
public class Llama2Runner {

    public static void main(String[] args) throws IOException {
        String modelPath = "models/stories15M.bin";
        File file = new File(modelPath);

        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel channel = raf.getChannel()) {

            // Zero-copy memory mapping
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            Config config = new Config(
                    buffer.getInt(), buffer.getInt(), buffer.getInt(), buffer.getInt(),
                    buffer.getInt(), buffer.getInt(), buffer.getInt()
            );

            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            TransformerWeights weights = TransformerWeights.load(config, floatBuffer);
            RunState state = new RunState(config);

            Tokenizer tokenizer;
            try {
                tokenizer = new Tokenizer("models/tokenizer.bin", config.vocab_size);
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }

            // Auto-regressive generation loop
            long generateStart = System.currentTimeMillis();
            int token = 1; // BOS token
            int pos = 0;
            int max_steps = 50;

            while (pos < max_steps) {
                Transformer.forward(config, weights, state, token, pos);

                // Greedy decoding
                int next_token = 0;
                float max_score = state.logits[0];
                for (int i = 1; i < config.vocab_size; i++) {
                    if (state.logits[i] > max_score) {
                        max_score = state.logits[i];
                        next_token = i;
                    }
                }

                if (next_token == 2 || next_token == 0) { // EOS token
                    break;
                }

                String piece = tokenizer.vocab[next_token];
                piece = piece.replace(" ", " ");
                piece = piece.replace("<0x0A>", "\n");

                System.out.print(piece);

                token = next_token;
                pos++;
            }

            long generateEnd = System.currentTimeMillis();
            System.out.printf("\n\nTokens generated: %d, Time: %d ms, Speed: %.2f tokens/s\n",
                    pos, (generateEnd - generateStart), (pos * 1000.0) / (generateEnd - generateStart));

            MathUtils.THREAD_POOL.shutdown();
        }
    }
}