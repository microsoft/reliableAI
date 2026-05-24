package ch.idsia.blip.core.inference.sample;


import ch.idsia.blip.core.Base;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public abstract class BaseSampler extends Base {

    protected long start;

    protected double max_time;

    protected double elapsed;

    protected int max_size = 100;

    protected short[] sample;

    protected int[] vars;

    protected int[] evid;

    public int n_var;

    protected Random rand;

    protected void prepare(double max_time, TIntIntHashMap evidence) {
        start = System.currentTimeMillis();
        this.max_time = max_time;

        // Already put evidence values in sample
        sample = new short[n_var];

        vars = evidence.keys();
        Arrays.sort(vars);
        evid = new int[vars.length];
        for (int i = 0; i < vars.length; i++) {
            int k = vars[i];

            evid[i] = evidence.get(k);
            sample[k] = (short) evid[i];
        }

        rand = getRandom();
    }

    protected boolean thereIsTime() {
        elapsed = ((System.currentTimeMillis() - start) / 1000.0);
        return elapsed < max_time;
    }

    public double PR(int v, int e, double max_time) {

        TIntIntHashMap values = new TIntIntHashMap();

        values.put(v, e);

        return PR(values, max_time);
    }

    /* Partition function and probability of evidence (PR inference) */
    public abstract double PR(TIntIntHashMap evidence, double max_time);

    /* Marginal probability distribution over a variable given evidence (MAR inference) */
    public abstract double[][] MAR(TIntIntHashMap evidence, double max_time);

    /* Computing the most likely assignment to all variables given evidence */
    public abstract short[] MAP(TIntIntHashMap evidence, double max_time);

    /* Computing the most likely assignment to a subset of variables given evidence */
    public abstract short[] MMAP(TIntIntHashMap evidence, int[] query, double max_time);

    public void writeMARoutput(String h, double[][] p_x_w) throws IOException {
        Writer w = getWriter(h + ".MAR2");

        wf(w, "MAR\n");
        wf(w, "%d ", p_x_w.length);
        for (int i = 0; i < p_x_w.length; i++) {
            wf(w, "\n%d ", p_x_w[i].length);
            for (int j = 0; j < p_x_w[i].length; j++) {
                wf(w, "%.7f ", p_x_w[i][j]);
            }
        }
        w.close();
    }

    public TIntIntHashMap getEvidence(String s) throws IOException {
        BufferedReader r = getReader(s);
        String l;
        String cnt = "";

        while ((l = r.readLine()) != null) {
            cnt += l + " ";
        }
        String[] g = cnt.trim().split("\\s+");
        TIntIntHashMap evid = new TIntIntHashMap();

        if (g.length <= 1) {
            return evid;
        }
        int n = Integer.valueOf(g[0]);

        for (int i = 0; i < n; i++) {
            evid.put(Integer.valueOf(g[1 + 2 * i]),
                    Integer.valueOf(g[2 + 2 * i]));
        }
        return evid;
    }
}
