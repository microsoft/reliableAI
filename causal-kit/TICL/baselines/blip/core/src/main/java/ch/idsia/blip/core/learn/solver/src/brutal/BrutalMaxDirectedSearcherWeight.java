package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.data.ArrayUtils.*;
import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class BrutalMaxDirectedSearcherWeight extends BrutalGreedySearcher {

    private TIntHashSet todo;

    private Result[] bests;

    private double[] minSk;
    private double[] maxSk;

    private int[][] parents;

    public BrutalMaxDirectedSearcherWeight(BaseSolver solver, int tw) {
        super(solver, tw);
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        minSk = new double[n_var];
        maxSk = new double[n_var];

        for (int i = 0; i < n_var; i++) {
            int j = m_scores[i].length - 1;

            minSk[i] = m_scores[i][j].sk;
            maxSk[i] = m_scores[i][0].sk;
        }

        parents = new int[n_var][];

        for (int i = 0; i < n_var; i++) {
            TIntHashSet l = new TIntHashSet();

            for (ParentSet ps : scores[i]) {
                for (int p : ps.parents) {
                    l.add(p);
                }
            }

            parents[i] = l.toArray();
            Arrays.sort(parents[i]);
        }
    }

    // Maximize a network!
    @Override
    public ParentSet[] search() {

        this.vars = smp.sample();

        // clear all
        clear();

        chooseClique();

        // p(Arrays.toString(vars));

        // Prepare structures for best handlers selection
        initCand();

        // Init the first maximal clique
        initClique();

        Result res;

        while (!todo.isEmpty()) {

            // Get the best subset of neighborhood - existing handle
            res = sampleWeighted();

            done(res.v);

            finalize(res);

            solver.checkTime();
            if (!solver.still_time) {
                return null;
            }
        }

        // check();

        return new_str;
    }

    private void chooseClique() {
        int theChosen = randInt(0, n_var - 1);

        vars = new int[n_var];

        TIntHashSet init = new TIntHashSet();

        init.add(theChosen);
        int[] cands = cloneArray(parents[theChosen]);

        Arrays.sort(cands);

        // If the chosen has not enough neighbours
        while (init.size() < tw + 1) {
            // Add random neighbouring variable
            int newChosen;

            if (cands.length > 0) {
                newChosen = cands[randInt(0, cands.length - 1)];
                cands = reduceArray(cands, newChosen);
            } else {
                newChosen = randInt(0, n_var - 1);
                while (init.contains(newChosen)) {
                    newChosen = randInt(0, n_var - 1);
                }
            }

            init.add(newChosen);
            // Add parents to cands
            for (int p : parents[newChosen]) {
                if (!init.contains(p) && !find(p, cands)) {
                    cands = expandArray(cands, p);
                }
            }
        }

        cloneArray(init.toArray(), vars, tw + 1);

        // pf("INITIAL CLIQUE: %s \n", Arrays.toString(vars));
    }

    private void done(int v) {
        todo.remove(v);
        if (bests[v] == null) {
            p("cdfjds");
        }
        // totWeight -= tryFirst[v].sk;
        bests[v] = null;
    }

    private void initCand() {
        // Init list of variables to evaluate
        todo = new TIntHashSet();
        bests = new Result[n_var];

        for (int v = 0; v < n_var; v++) {
            todo.add(v);
            Result r = new Result(v, m_scores[v][m_scores[v].length - 1],
                    new SIntSet(), -1);

            bests[v] = r;
        }
    }

    @Override
    protected void initClique() {
        // Initial handler: best DAG with (1..tw+1) variables
        initCl = new int[tw + 1];
        System.arraycopy(vars, 0, initCl, 0, tw + 1);

        // Find best, do some asobs iterations
        ParentSet[] best_str = exploreAll();

        // Update parent se
        for (int v : initCl) {
            update(v, best_str[v]);
            done(v);
        }

        // Add new handlers
        ArrayList<SIntSet> l_a = new ArrayList<SIntSet>();

        for (int v : initCl) {
            SIntSet s = new SIntSet(reduceArray(initCl, v));

            addHandler(s);
            l_a.add(s);
        }

        // Add new handlers
        updateBests(l_a);
    }

    private void finalize(Result res) {
        // update the chosen parent set
        update(res.v, res.ps);

        // add the new handlers
        SIntSet orig = res.handle;
        ArrayList<SIntSet> l_a = new ArrayList<SIntSet>();

        for (int elim : orig.set) {
            SIntSet s = new SIntSet(
                    reduceAndIncreaseArray(orig.set, res.v, elim));

            addHandler(s);
            l_a.add(s);
        }

        updateBests(l_a);
    }

    private void updateBests(ArrayList<SIntSet> l_a) {
        // Update best handlers
        TIntIterator it = todo.iterator();

        while (it.hasNext()) {
            int v = it.next();

            Pair<ParentSet, SIntSet> r = bestParentSet(v, l_a);

            if (r == null) {
                continue;
            }

            ParentSet bestPset = r.getFirst();
            SIntSet handler = r.getSecond();

            double sk = (minSk[v] - bestPset.sk) / (minSk[v] - maxSk[v]);

            if (sk > bests[v].sk) {
                // totWeight -= tryFirst[v].sk;
                Result c = new Result(v, bestPset, handler, sk);

                bests[v] = c;
                // totWeight += tryFirst[v].sk;
            }
        }
    }

    private Pair<ParentSet, SIntSet> bestParentSet(int v, ArrayList<SIntSet> l_a) {
        for (ParentSet p : m_scores[v]) {

            for (SIntSet h : l_a) {
                if (containsAll(p.parents, h.set)) {
                    return new Pair<ParentSet, SIntSet>(p, h);
                }
            }
        }

        return null;
    }

    protected class Result {

        public double sk;
        SIntSet handle;
        public int v;
        public ParentSet ps;

        Result(int v, ParentSet ps, SIntSet handle, double sk) {
            this.ps = ps;
            this.v = v;
            this.handle = handle;
            this.sk = sk;
        }

        @Override
        public String toString() {
            return f("%d %s %s %.4f", v, ps.toString(),
                    Arrays.toString(handle.set), sk);
        }
    }

    private Result sampleWeighted() {

        double r = solver.randDouble() - Math.pow(2, -10);
        int sel = -1;

        double totWeight = 0;

        for (int v = 0; v < n_var; v++) {
            if (bests[v] == null) {
                continue;
            }

            if (bests[v].sk >= 0.95) {
                return bests[v];
            }

            totWeight += bests[v].sk;
        }

        for (int v = 0; v < n_var && sel == -1; v++) {
            if (bests[v] == null) {
                continue;
            }

            double s = bests[v].sk / totWeight;

            if (r <= s) {
                sel = v;
            }
            r -= s;
        }

        return bests[sel];
    }

    private void check() {

        BayesianNetwork b = new BayesianNetwork(new_str);

        try {
            b.checkAcyclic();
        } catch (CyclicGraphException e) {
            solver.log("WHHHHHAAAAT");
        }
    }

}
