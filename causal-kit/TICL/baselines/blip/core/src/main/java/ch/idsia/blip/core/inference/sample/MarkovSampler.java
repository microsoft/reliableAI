package ch.idsia.blip.core.inference.sample;


import ch.idsia.blip.core.utils.MarkovNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import java.util.Arrays;
import java.util.HashMap;


public class MarkovSampler extends BaseSampler {

    private final MarkovNetwork mn;

    private int cnt_new;

    private int cnt_start;

    public MarkovSampler(MarkovNetwork mn) {
        this.mn = mn;
        this.n_var = mn.n_var;
        mn.updateCliqueAssignments();
    }

    @Override
    public short[] MMAP(TIntIntHashMap evidence, int[] query, double max_time) {

        prepare(max_time, evidence);

        // Prepare non-evidence variables to change
        TIntArrayList aux_freedom = new TIntArrayList();

        for (int i = 0; i < mn.n_var; i++) {
            if (!find(i, vars)) {
                aux_freedom.add(i);
            }
        }
        int[] freedom = aux_freedom.toArray();

        short[] sol;

        HashMap<MpeSol, Integer> cache = new HashMap<MpeSol, Integer>();

        // Loop until we have time
        while (thereIsTime()) {

            if (sample(freedom)) {
                continue;
            }

            sol = new short[query.length];
            for (int i = 0; i < query.length; i++) {
                sol[i] = sample[query[i]];
            }

            MpeSol s = new MpeSol(sol);

            if (cache.containsKey(s)) {
                cache.put(s, cache.get(s) + 1);
            } else if (cache.size() < max_size) {
                cache.put(s, 1);
            }
            // Update in cache count
        }

        int max = 0;
        MpeSol best = null;

        for (MpeSol s : cache.keySet()) {
            // pf("%s %d \n", Arrays.toString(s.set), cache.get(s));
            if (cache.get(s) > max) {
                max = cache.get(s);
                best = s;
            }
        }

        return best.set;
    }

    @Override
    public short[] MAP(TIntIntHashMap evidence, double max_time) {

        prepare(max_time, evidence);

        // Prepare non-evidence variables to change
        TIntArrayList aux_freedom = new TIntArrayList();

        for (int i = 0; i < mn.n_var; i++) {
            if (!find(i, vars)) {
                aux_freedom.add(i);
            }
        }
        int[] freedom = aux_freedom.toArray();

        HashMap<MpeSol, Integer> cache = new HashMap<MpeSol, Integer>();

        // Loop until we have time
        while (thereIsTime()) {

            if (sample(freedom)) {
                continue;
            }

            MpeSol s = new MpeSol(sample);

            if (cache.containsKey(s)) {
                cache.put(s, cache.get(s) + 1);
            } else if (cache.size() < max_size) {
                cache.put(s, 1);
            }
            // Update in cache count
        }

        int max = 0;
        MpeSol best = null;

        for (MpeSol s : cache.keySet()) {
            // pf("%s %d \n", Arrays.toString(s.set), cache.get(s));
            if (cache.get(s) > max) {
                max = cache.get(s);
                best = s;
            }
        }

        return best.set;
    }

    private boolean sample(int[] freedom) {
        if (cnt_new == 0) {
            ShuffleArray(freedom);

            for (int v : freedom) {
                sample[v] = (short) rand.nextInt(mn.l_ar_var[v]);
            }

            cnt_new = 1000;
            cnt_start = 100;
        }

        for (int v : freedom) {
            // Resample it
            mn.resample(sample, v, rand);
        }

        // (1) Wait a little when you jump to a new random start
        // Or (2) wait a little between different sampling
        if (cnt_start == 0) {
            cnt_start = 10;
        } else {
            cnt_start--;
            cnt_new--;
            return true;
        }
        return false;
    }

    @Override
    public double[][] MAR(TIntIntHashMap evidence, double max_time) {

        prepare(max_time, evidence);

        // Prepare non-evidence variables to change
        TIntArrayList aux_freedom = new TIntArrayList();

        for (int i = 0; i < mn.n_var; i++) {
            if (!find(i, vars)) {
                aux_freedom.add(i);
            }
        }
        int[] freedom = aux_freedom.toArray();

        int tot = 0;

        double[][] cnt = new double[mn.n_var][];

        for (int i = 0; i < mn.n_var; i++) {
            cnt[i] = new double[mn.l_ar_var[i]];
        }

        // Loop until we have time
        while (thereIsTime()) {

            if (sample(freedom)) {
                continue;
            }

            // Finally, sample!
            for (int i = 0; i < mn.n_var; i++) {
                cnt[i][sample[i]] += 1;
            }

            tot += 1;

            // p(Arrays.toString(sample));
        }

        for (int i = 0; i < mn.n_var; i++) {
            for (int v = 0; v < mn.l_ar_var[i]; v++) {
                cnt[i][v] /= tot;
            }
        }

        return cnt;
    }

    @Override
    public double PR(TIntIntHashMap evidence, double max_time) {

        prepare(max_time, evidence);

        int[] ord = new int[n_var];

        for (int i = 0; i < mn.n_var; i++) {
            ord[i] = i;
        }

        // Prepare evidence values to compare
        int[] vars = evidence.keys();

        Arrays.sort(vars);
        int[] res = new int[vars.length];

        for (int i = 0; i < vars.length; i++) {
            res[i] = evidence.get(vars[i]);
        }

        boolean ok;
        int cnt = 0;
        int tot = 0;

        while (thereIsTime()) {

            if (sample(ord)) {
                continue;
            }

            // Update counters
            ok = true;
            for (int i = 0; i < this.vars.length && ok; i++) {
                ok = (sample[this.vars[i]] == res[i]);
            }

            if (ok) {
                cnt += 1;
            }

            tot += 1;

        }

        return cnt * 1.0 / tot;
    }

    @Override
    protected void prepare(double max_time, TIntIntHashMap evidence) {
        super.prepare(max_time, evidence);

        // Counter for new random restart
        cnt_new = 0;

        // Counter between sampling
        cnt_start = 100;
    }

    private void ShuffleArray(int[] array) {
        int index, temp;

        for (int i = array.length - 1; i > 0; i--) {
            index = rand.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
}
