package com.kirisu.llama2;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;

/**
 * Zero-dependency math and neural network operators.
 */
public class MathUtils {

    private static final int CORES = Runtime.getRuntime().availableProcessors();
    public static final ExecutorService THREAD_POOL = Executors.newFixedThreadPool(CORES);

    public static void rmsnorm(float[] o, float[] x, float[] weight, int size) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = (float)(1.0f / Math.sqrt(ss));
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }

    /**
     * Parallel matrix multiplication using JUC.
     */
    public static void matmul(float[] xout, float[] x, float[] w, int n, int d) {
        if (d < CORES) {
            for (int i = 0; i < d; i++) {
                float val = 0.0f;
                for (int j = 0; j < n; j++) {
                    val += w[i * n + j] * x[j];
                }
                xout[i] = val;
            }
            return;
        }

        CountDownLatch latch = new CountDownLatch(CORES);
        int chunkSize = d / CORES;

        for (int c = 0; c < CORES; c++) {
            final int coreId = c;
            THREAD_POOL.submit(() -> {
                try {
                    int startRow = coreId * chunkSize;
                    int endRow = (coreId == CORES - 1) ? d : startRow + chunkSize;

                    for (int i = startRow; i < endRow; i++) {
                        float val = 0.0f;
                        int weightOffset = i * n;
                        for (int j = 0; j < n; j++) {
                            val += w[weightOffset + j] * x[j];
                        }
                        xout[i] = val;
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public static void softmax(float[] x, int size) {
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = (float)Math.exp(x[i] - max_val);
            sum += x[i];
        }
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    public static void silu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            float val = x[i];
            val = val / (1.0f + (float) Math.exp(-val));
            x[i] = val;
        }
    }
}