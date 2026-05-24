package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.Base;
import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.learn.solver.brtl.BrutalUndirectedSolver;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceAndIncreaseArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * Brutal on Undirected
 */
public class BrutalUndirectedSearcher extends Base implements Searcher {

    protected final BrutalUndirectedSolver solver;

    protected final int tw;

    protected final Und und;

    protected final int n_var;
    protected final int thread;

    private final Sampler smp;

    protected int[] vars;

    protected Und new_und;

    protected int new_sk;

    protected TreeSet<SIntSet> handles;

    protected int[] initCl;

    protected long start;

    protected int cnt;

    public BrutalUndirectedSearcher(BrutalUndirectedSolver solver, int tw, Und und) {
        this.solver = solver;
        this.tw = tw;
        this.und = und;

        this.n_var = und.n;

        this.thread = solver.thread;
        solver.thread += 1;

        this.smp = solver.getSampler();
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {// does nothing with scores
    }

    // Greedily optimize a network!
    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        start = System.currentTimeMillis();
        cnt = 1;

        // clear all
        clear();

        // Init the first maximal clique
        initClique();

        Result res;

        // Greedy behaviour (follow sampling)
        for (int i = tw + 1; i < n_var; i++) {
            int v = vars[i];

            res = bestSubset(v);

            // pf("Chosen %d \n", res.v);
            finalize(res);

            solver.checkTime();
            if (!solver.still_time) {
                return null;
            }

        }

        solver.newUndirected(new_sk, new_und);

        return null;
    }

    protected void finalize(Result res) {
        // update the chosen parent set
        update(res.v, res.neigh);

        // add the new handlers
        // add new handler = new clique with size tw
        // created  just now
        SIntSet h = res.handle;

        for (int elim : h.set) {
            addHandler(new SIntSet(reduceAndIncreaseArray(h.set, res.v, elim)));
        }
    }

    protected void update(int v, int[] fin) {
        for (int f : fin) {
            new_und.mark(v, f);
            new_sk += 1;
            // pf("Marking %d %d \n", v, f);

        }
    }

    protected void initClique() {
        initCl = new int[tw + 1];
        System.arraycopy(vars, 0, initCl, 0, tw + 1);

        // Update parent set
        for (int i1 = 0; i1 < initCl.length; i1++) {
            int v1 = initCl[i1];

            for (int i2 = i1 + 1; i2 < initCl.length; i2++) {
                int v2 = initCl[i2];

                if (find(v1, und.neigh[v2])) {
                    new_und.mark(v1, v2);
                    new_sk += 1;
                    // pf("Marking %d %d \n", v1, v2);
                }
            }
        }

        // Add new handlers
        for (int v : initCl) {
            addHandler(new SIntSet(reduceArray(initCl, v)));
        }
    }

    protected void clear() {

        new_und = new Und(und.n);
        new_sk = 0;

        // List of handlers for parent sets
        // (list of quasi-maximal cliques, where a variable
        // can choose its parent sets)
        handles = new TreeSet<SIntSet>();

        start = System.currentTimeMillis();
        cnt = 1;
    }

    protected void addHandler(SIntSet sIntSet) {
        handles.add(sIntSet);
    }

    protected Result bestSubset(int v) {

        int max_sk = 0;
        SIntSet max_h = null;

        for (SIntSet h : handles) {

            int sk = getSk(und.neigh[v], h.set);

            if (sk > max_sk) {
                max_h = h;
                max_sk = sk;
            }
        }

        if (max_sk == 0) {
            return new Result(v, new int[0], rand(handles));
        }

        int[] fin = ArrayUtils.intersect(und.neigh[v], max_h.set);

        return new Result(v, fin, max_h);
    }

    protected int randInt(int a, int b) {
        return solver.randInt(a, b);
    }

    protected int getSk(int[] neigh, int[] set) {
        return ArrayUtils.intersectN(neigh, set);
    }

    protected SIntSet rand(TreeSet<SIntSet> h) {
        int v = solver.randInt(0, h.size());
        Iterator<SIntSet> i = h.iterator();

        while (v > 1) {
            i.next();
            v--;
        }
        return i.next();
    }

    protected class Result implements Comparable<Result> {

        public SIntSet handle;
        public int v;
        public int[] neigh;

        public Result(int v, int[] neigh, SIntSet handle) {
            this.neigh = neigh;
            this.v = v;
            this.handle = handle;
        }

        @Override
        public int compareTo(Result o) {
            if (neigh.length < o.neigh.length) {
                return 1;
            }
            if (neigh.length > o.neigh.length) {
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
            return f("%d %s %s", v, Arrays.toString(neigh),
                    Arrays.toString(handle.set));
        }
    }
}
