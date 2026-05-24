package ch.idsia.blip.core.learn.solver.samp;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.util.Random;


public class EntropyBSampler extends EntropySampler {

    protected double[] weight_r;

    private TIntArrayList vars;

    @Override
    public int[] sample() {
        double v = r.nextDouble();

        if (v > 0.66) {
            // p("direct");
            return sampleWeighted(n, weight);

        }

        if (v > 0.33) {
            // p("inverse");
            return sampleWeighted(n, weight_r);
        }

        // p("normal");
        return sampleSimple();
    }

    private int[] sampleSimple() {
        int[] nv;

        synchronized (lock) {
            vars.shuffle(r);
            nv = vars.toArray().clone();
        }

        return nv;
    }

    public EntropyBSampler(String ph_dat, int n, Random r) {
        super(ph_dat, n, r);
    }

    @Override
    public void init() {
        super.init();

        weight_r = new double[n];
        for (int i = 0; i < n; i++) {
            weight_r[i] = 1.0 / weight[i];
        }

        vars = new TIntArrayList();
        for (int i = 0; i < n; i++) {
            vars.add(i);
        }
    }
}
