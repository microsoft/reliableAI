package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.other.Clique;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.*;
import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class BrutalMaxDirectedSearcher extends BrutalOldSearcher {

    private TIntHashSet todo;

    private TreeSet<Result> cand;

    private Result[] bests;

    private double[] minSk;
    private double[] maxSk;

    private int[][] parents;

    public List<Clique> junctTree;

    public BrutalMaxDirectedSearcher(BaseSolver solver, int tw) {
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

    @Override
    protected void clear() {
        super.clear();

        junctTree = new ArrayList<Clique>();
    }

    // Maximize a network!
    @Override
    public ParentSet[] search() {

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
            res = cand.pollFirst();

            done(res.v);

            // pf("Chosen %d - %s\n", res.v, todo.toString());

            // p(res);
            // p(todo);

            // pf("Chosen %d \n", res.v);
            finalize(res);

            solver.checkTime();
            if (!solver.still_time) {
                return null;
            }

        }

        // check();

        // grSolver.propose(new_sk, new_str, junctTree);

        return new_str;
    }

    private void check() {

        BayesianNetwork b = new BayesianNetwork(new_str);

        try {
            b.checkAcyclic();
        } catch (CyclicGraphException e) {
            solver.log("WHHHHHAAAAT");
        }
    }

    private void chooseClique() {

        int theChosen = randInt(0, n_var - 1);

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

        initCl = init.toArray();
        Arrays.sort(initCl);

        // pf("INITIAL CLIQUE: %s \n", Arrays.toString(vars));
    }

    private void done(int v) {
        todo.remove(v);
        if (bests[v] == null) {
            p("cdfjds");
        }
        cand.remove(bests[v]);
        bests[v] = null;
    }

    private void initCand() {
        // Init list of variables to evaluate
        todo = new TIntHashSet();
        cand = new TreeSet<Result>();
        bests = new Result[n_var];

        for (int v = 0; v < n_var; v++) {
            todo.add(v);
            Result r = new Result(v, m_scores[v][m_scores[v].length - 1],
                    new SIntSet(), 1, null);

            cand.add(r);
            bests[v] = r;
        }
    }

    @Override
    // Initial handler: best DAG with (1..tw+1) variables
    protected void initClique() {

        // Find best, do some asobs iterations
        ParentSet[] best_str = exploreAll();

        // Update parent se
        for (int v : initCl) {
            update(v, best_str[v]);
            done(v);
        }

        Clique cl = new Clique(initCl, null);

        junctTree.add(cl);

        // Add new handlers
        ArrayList<SIntSet> l_a = new ArrayList<SIntSet>();

        for (int v : initCl) {
            SIntSet s = new SIntSet(reduceArray(initCl, v));

            addHandler(s);
            l_a.add(s);
        }
        updateBests(l_a, cl);
    }

    protected void finalize(Result res) {
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

        int[] pars = expandArray(res.handle.set, res.v);
        Clique cl = new Clique(pars, res.cl);

        junctTree.add(cl);

        updateBests(l_a, cl);
    }

    private void updateBests(ArrayList<SIntSet> l_a, Clique cl) {
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

            double sk = (bestPset.sk - maxSk[v]) / (minSk[v] - maxSk[v]);

            if (sk < bests[v].sk) {
                // int ct = cand.size();
                cand.remove(bests[v]);
                Result c = new Result(v, bestPset, handler, sk, cl);

                cand.add(c);
                bests[v] = c;
                // if (cand.size() != ct)
                // p("Jkdjfdkfjds!!");
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

    protected class Result implements Comparable<Result> {

        public double sk;
        public SIntSet handle;
        public int v;
        public ParentSet ps;
        public Clique cl;

        public Result(int v, ParentSet ps, SIntSet handle, double sk, Clique cl) {
            this.ps = ps;
            this.v = v;
            this.handle = handle;
            this.sk = sk;
            this.cl = cl;
        }

        @Override
        public int compareTo(Result o) {
            if (sk > o.sk) {
                return 1;
            }
            if (sk < o.sk) {
                return -1;
            }

            if (v < o.v) {
                return 1;
            }
            if (v > o.v) {
                return -1;
            }

            if (equals(o)) {
                return 0;
            } else {
                return -1;
            }
        }

        // @Override
        // public boolean equals(Object o) {
        // if (this == o) return true;
        // if (o == null || getClass() != o.getClass()) return false;
        //
        // Result result = (Result) o;
        //
        // if (v != result.v) return false;
        // if (handle != null ? !handle.equals(result.handle) : result.handle != null) return false;
        // return Arrays.equals(neigh, result.neigh);
        // }
        //
        // @Override
        // public int hashCode() {
        // int result = handle != null ? handle.hashCode() : 0;
        // result = 31 * result + v;
        // result = 31 * result + Arrays.hashCode(neigh);
        // return result;
        // }

        @Override
        public String toString() {
            return f("%d %s %s %.4f", v, ps.toString(),
                    Arrays.toString(handle.set), sk);
        }
    }
}
