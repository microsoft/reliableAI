package ch.idsia.blip.core.learn.solver.samp;


import ch.idsia.blip.core.utils.data.ArrayUtils;

import java.util.Random;


public class SimpleSampler implements Sampler {

    private final int n;
    private final Random r;

    private Object lock = new Object();

    private int[] nv;

    @Override
    public int[] sample() {

        synchronized (lock) {
            for (int j = 0; j < Math.max(3, nv.length / 10); j++) {
                for (int i = nv.length; i-- > 1;) {
                    ArrayUtils.swap(nv, i, r.nextInt(i));
                }
            }
        }

        return nv;
    }

    public SimpleSampler(int n, Random r) {
        this.n = n;
        this.r = r;

        nv = new int[n];
        for (int i = 0; i < n; i++) {
            nv[i] = i;
        }
    }

    @Override
    public void init() {}
}
