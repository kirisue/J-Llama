package com.kirisu.llama2;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Basic byte-pair encoding (BPE) decoder.
 */
public class Tokenizer {
    public final String[] vocab;
    public final float[] vocab_scores;
    public final int max_token_length;

    public Tokenizer(String filepath, int vocab_size) throws Exception {
        vocab = new String[vocab_size];
        vocab_scores = new float[vocab_size];

        File file = new File(filepath);
        if (!file.exists()) {
            throw new RuntimeException("Tokenizer file not found.");
        }

        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            max_token_length = buffer.getInt();

            for (int i = 0; i < vocab_size; i++) {
                vocab_scores[i] = buffer.getFloat();
                int len = buffer.getInt();
                byte[] bytes = new byte[len];
                buffer.get(bytes);
                vocab[i] = new String(bytes, "UTF-8");
            }
        }
    }
}