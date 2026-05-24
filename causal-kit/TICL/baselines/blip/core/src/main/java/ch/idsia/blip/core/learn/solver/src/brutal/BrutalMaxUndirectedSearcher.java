package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.learn.solver.brtl.BrutalUndirectedSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;

import static ch.idsia.blip.core.utils.data.ArrayUtils.*;
import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class BrutalMaxUndirectedSearcher extends BrutalUndirectedSearcher {

    protected TIntHashSet todo;

    private double totWeight = 0;

    protected Result[] bests;

    public BrutalMaxUndirectedSearcher(BrutalUndirectedSolver solver, int tw, Und und) {
        super(solver, tw, und);
    }

    // Maximize a network!
    @Override
    public ParentSet[] search() {

        // clear all
        clear();

        // Choose initial clique
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

            // pf("Chosen %d - %s\n", res.v, todo.toString());

            // p(res);
            // p(todo);

            if (res.neigh.length == 0) {
                pf("Thread %d, null neighborhood!!! \n", thread);
                return null;
            }
            // pf("Chosen %d \n", res.v);
            finalize(res);

            solver.checkTime();
            if (!solver.still_time) {
                return null;
            }

            double elapsed = (System.currentTimeMillis() - start) / 1000.0;

            if (elapsed > cnt * 10) {
                pf("Thread %d, remaining to look: %d, score: %d - %.2f \n",
                        thread, todo.size(), new_sk, elapsed);
                cnt += 1;
            }
        }

        solver.newUndirected(new_sk, new_und);

        return null;
    }

    protected void done(int v) {
        todo.remove(v);
        if (bests[v] == null) {
            p("cdfjds");
        }
        totWeight -= bests[v].neigh.length;
        bests[v] = null;
    }

    private void chooseClique() {
        int theChosen = randInt(0, n_var - 1);

        vars = new int[n_var];

        int[] copy = new int[und.neigh[theChosen].length];

        cloneArray(und.neigh[theChosen], copy);

        // pf("Chosen: %d \n", theChosen);

        // If the chosen has not enough neighbours
        if (copy.length < tw) {
            // Add random variables until we have enough
            while (copy.length < tw) {
                // Select a new element on the initial cliques
                int ix = copy[randInt(0, copy.length - 1)];
                int[] neigh = und.neigh[ix];
                // Select a new random element from its neighborhood
                int ix2 = neigh[randInt(0, neigh.length - 1)];

                if (!find(ix2, copy) && ix2 != theChosen) {
                    copy = expandArray(copy, ix2);
                }

                // p(Arrays.toString(copy));
            }
        } else {
            // Select a random subset of the neighborhood
            ArrayUtils.shuffleArray(copy, solver.rand);
        }

        cloneArray(copy, vars, tw);
        vars[tw] = theChosen;

        // pf("INITIAL CLIQUE: %s \n", Arrays.toString(vars));
    }

    private void initCand() {
        // Init list of variables to evaluate
        todo = new TIntHashSet();
        bests = new Result[n_var];

        for (int i = 0; i < n_var; i++) {
            todo.add(i);
            Result r = new Result(i, new int[0], new SIntSet());

            bests[i] = r;
        }
    }

    @Override
    protected void initClique() {
        initCl = new int[tw + 1];
        System.arraycopy(vars, 0, initCl, 0, tw + 1);

        // p(Arrays.toString(initCl));

        // Update parent set
        for (int i1 = 0; i1 < initCl.length; i1++) {
            int v1 = initCl[i1];

            done(v1);
            for (int i2 = i1 + 1; i2 < initCl.length; i2++) {
                int v2 = initCl[i2];

                if (find(v1, und.neigh[v2])) {
                    new_und.mark(v1, v2);
                    new_sk += 1;
                    // pf("Initial marking %d %d \n", v1, v2);
                }
            }
        }

        // Add new handlers
        ArrayList<SIntSet> l_a = new ArrayList<SIntSet>();

        for (int v : initCl) {
            SIntSet s = new SIntSet(reduceArray(initCl, v));

            addHandler(s);
            l_a.add(s);
        }

        updateBests(l_a);
    }

    private void finalize(Result res) {
        // update the chosen parent set
        update(res.v, res.neigh);

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

    protected void updateBests(ArrayList<SIntSet> l_a) {
        // Update best handlers
        TIntIterator it = todo.iterator();

        while (it.hasNext()) {
            int v = it.next();

            int sk = bests[v].neigh.length;

            for (SIntSet h : l_a) {
                int[] neigh = ArrayUtils.intersect(und.neigh[v], h.set);

                if (neigh.length > sk) {
                    // TODO remove
                    totWeight -= bests[v].neigh.length;
                    Result c = new Result(v, neigh, h);

                    bests[v] = c;
                    totWeight += bests[v].neigh.length;
                    sk = bests[v].neigh.length;

                }
            }
        }
    }

    protected Result sampleWeighted() {

        double r = solver.randDouble() - Math.pow(2, -10);
        int sel = -1;

        for (int v = 0; v < n_var && sel == -1; v++) {
            if (bests[v] == null) {
                continue;
            }

            double s = bests[v].neigh.length / totWeight;

            if (s > 0 && r <= s) {
                sel = v;
            }
            r -= s;
        }

        return bests[sel];
    }

    protected class Result {

        public SIntSet handle;
        public int v;
        public int[] neigh;

        public Result(int v, int[] neigh, SIntSet handle) {
            this.neigh = neigh;
            this.v = v;
            this.handle = handle;
        }
    }

}
