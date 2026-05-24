package ch.idsia.blip.core.learn.solver.samp;


import java.util.Random;


public class MIBSampler extends MISampler {

    protected double[] weight_r;

    int turn;

    @Override
    public int[] sample() {

        if (turn == 0) {
            turn += 1;
            return sampleWeighted(n, weight);
        } else if (turn == 1) {
            turn += 1;
            return sampleWeighted(n, weight_r);
        } else {
            turn = 0;
            return sample();
        }

    }

    public MIBSampler(String ph_dat, int n, Random r) {
        super(ph_dat, n, r);
    }

    @Override
    public void init() {
        super.init();

        weight_r = new double[n];
        for (int i = 0; i < n; i++) {
            weight_r[i] = 1.0 / weight[i];
        }
    }
}
