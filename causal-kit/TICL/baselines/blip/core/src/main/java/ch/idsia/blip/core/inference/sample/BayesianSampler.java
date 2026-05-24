package ch.idsia.blip.core.inference.sample;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import java.util.Arrays;
import java.util.HashMap;


public class BayesianSampler extends BaseSampler {

    private final BayesianNetwork bn;

    private SamGe s;

    private int[] ord;

    public BayesianSampler(BayesianNetwork bn) {
        this.bn = bn;
        this.n_var = bn.n_var;
        this.s = new SamGe();
    }

    /* Computing the most likely assignment to all variables given evidence */
    public short[] MAP(TIntIntHashMap evidence, double max_time) {

        prepare(max_time, evidence);

        // Get sampling order (don't put evidence variables)
        ord = getSampleOrderMAR(bn, vars);

        HashMap<MpeSol, Integer> cache = new HashMap<MpeSol, Integer>();

        while (thereIsTime()) {

            sample();

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

    private void sample() {
        for (int n : ord) {
            sample[n] = s.getVariableSample(n, sample);
        }
    }

    @Override
    public short[] MMAP(TIntIntHashMap evidence, int[] query, double max_time) {
        prepare(max_time, evidence);

        // Get sampling order (don't put evidence variables)
        ord = getSampleOrderMAR(bn, vars);

        short[] sol;

        HashMap<MpeSol, Integer> cache = new HashMap<MpeSol, Integer>();

        while (thereIsTime()) {

            sample();

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

    /* Marginal probability distribution over a variable given evidence (MAR inference) */
    public double[][] MAR(TIntIntHashMap evidence, double max_time) {
        prepare(max_time, evidence);

        // Likelihood weighting of evidence Pr(e)
        double P_w = 0;

        // Likelihood estimate of each Pr(x|e)
        double[][] P_x_w = new double[bn.n_var][];

        for (int i = 0; i < bn.n_var; i++) {
            P_x_w[i] = new double[bn.arity(i)];
        }

        // Get sampling order (don't put evidence variables)
        ord = getSampleOrderMAR(bn, vars);

        int cnt = 0;

        // Loop until we have time
        while (thereIsTime()) {

            sample();

            // Get likelihood (product of all network paramaters of evidence)
            double w = 1;

            for (int i = 0; i < vars.length; i++) {
                int v = vars[i];
                double[] pt = bn.potentials(v);
                int ix = bn.potentialIndex(v, sample);
                int j = ix * bn.arity(v);

                j += evid[i];
                double p = pt[j];

                w *= p;
            }

            // Update likelihood of evidence
            P_w += w;

            // Update likelihood of sampled values
            for (int i = 0; i < bn.n_var; i++) {
                P_x_w[i][sample[i]] += w;
            }

            cnt += 1;
        }

        // Weight likelihood
        for (int i = 0; i < bn.n_var; i++) {
            for (int j = 0; j < bn.arity(i); j++) {
                P_x_w[i][j] /= P_w;
            }
        }

        double p_w = P_w / cnt;

        return P_x_w;
    }

    /* Partition function and probability of evidence (PR inference) */
    public double PR(TIntIntHashMap evidence, double max_time) {

        prepare(max_time, evidence);

        int[] res = new int[vars.length];

        for (int i = 0; i < vars.length; i++) {
            res[i] = evidence.get(vars[i]);
        }

        ord = getSampleOrderPR();

        int tot = 0;
        double cnt = 0;

        boolean ok;

        while (thereIsTime()) {

            sample();

            ok = true;
            for (int i = 0; i < vars.length && ok; i++) {
                ok = (sample[vars[i]] == res[i]);
            }

            if (ok) {
                cnt += 1;
            }

            tot += 1;

            // p(Arrays.toString(sample).replace("[", "").replace("]", ""));
        }

        return cnt / tot;

    }

    private int[] getSampleOrderPR() {
        TIntArrayList barren = findBarren(bn, vars);
        int[] ord = bn.getTopologicalOrder();
        TIntArrayList newOrd = new TIntArrayList();

        for (int o : ord) {
            if (!barren.contains(o)) {
                newOrd.add(o);
            }
        }
        return newOrd.toArray();
    }

    private int[] getSampleOrderMAR(BayesianNetwork bn, int[] vars) {
        int[] ord = bn.getTopologicalOrder();
        TIntArrayList newOrd = new TIntArrayList();

        for (int o : ord) {
            if (!find(o, vars)) {
                newOrd.add(o);
            }
        }
        return newOrd.toArray();
    }

    private TIntArrayList findBarren(BayesianNetwork bn, int[] query) {
        return findBarren(bn, query, new int[0]);
    }

    private TIntArrayList findBarren(BayesianNetwork bn, int[] query, int[] ev_keys) {

        TIntArrayList barren = new TIntArrayList();

        Arrays.sort(ev_keys);

        TIntArrayList rev_topol = new TIntArrayList();

        rev_topol.addAll(bn.getTopologicalOrder());

        rev_topol.reverse();

        for (int i = 0; i < rev_topol.size(); i++) {
            int v = rev_topol.get(i);

            if (isBarren(bn, query, ev_keys, barren, v)) {
                barren.add(v);
            }
        }

        barren.sort();

        return barren;

    }

    /**
     * Determine if the variable is barren in the current situation.
     *
     * @param query    variables of query
     * @param evidence evidence applied
     * @param barren   dand barren variables
     * @param v        variable to analyze
     * @return if the variable is barren w.r.t. query, evidence, and dand barren
     */
    private boolean isBarren(BayesianNetwork bn, int[] query, int[] evidence, TIntArrayList barren, int v) {
        if (find(v, query)) {
            return false;
        }
        if (find(v, evidence)) {
            return false;
        }
        for (int c = 0; c < bn.n_var; c++) {
            // If it's not a children of v continue
            if (Arrays.binarySearch(bn.parents(c), v) < 0) {
                continue;
            }
            // If it's a children, but it's not in barren, then v is not barren
            if (!barren.contains(c)) {
                return false;
            }
        }
        return true;
    }

    public int[] getQuery(String q) {
        String[] aux = q.split("\\s+");
        int n = Integer.valueOf(aux[0]);
        int[] query = new int[n];

        for (int i = 0; i < n; i++) {
            query[i] = Integer.valueOf(aux[i + 1]);
        }
        return query;
    }

    /**
     public double PR(BayesianNetwork bn, int v, int e) {
     SamGe s = new SamGe();
     s.bn = bn;
     int[] ord = bn.getTopologicalOrder();

     int tot = 10000;
     short e2 = (short) e;

     int cnt = 0;
     for (int i = 0; i < tot; i++) {
     short[] sample = new short[bn.n_var];

     for (int n : ord) {
     sample[n] = s.getVariableSample(n, sample);
     }

     if (sample[v] == e2)
     cnt += 1;

     // p(Arrays.toString(sample).replace("[", "").replace("]", ""));
     }

     return (cnt + 1.0) / (tot + 1.0);

     }*/
}
