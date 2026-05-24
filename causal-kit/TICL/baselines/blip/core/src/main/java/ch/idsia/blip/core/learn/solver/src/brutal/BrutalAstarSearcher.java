package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.Arrays;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceAndIncreaseArray;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BrutalAstarSearcher extends BrutalOldSearcher {

    // heuristic estimation of choices
    protected double[] heuStr;

    // list of states to explore
    TreeSet<State> open;

    // worst state to search
    private double worstQueueScore;

    private long queue_size = (long) Math.pow(2, 10);

    private int exp_limit = 0;

    public BrutalAstarSearcher(BaseSolver solver, int tw, int exp_limit) {
        super(solver, tw);
        this.exp_limit = exp_limit;
    }

    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        // init everything
        clear();

        // Build the vector with the heuristic estimation for each variable
        initHVector();

        // Init the first maximal clique
        initClique();

        // Create first state
        State st = createState(tw + 1, new_str, handles);

        open.add(st);

        boolean completed = false;

        // Repeat until "Human Optimality Project"
        while (!open.isEmpty() && solver.still_time) {
            // get new best state
            st = open.pollFirst();

            if (st.index >= n_var) {
                completed = true;
                break;
            }

            if (solver.verbose > 1) {
                solver.logf("considering state: %s \n", st);
            }

            // get new variable to work on
            int v = vars[st.index];

            TreeSet<State> candidates = new TreeSet<State>();

            // for every handler available in the state
            for (SIntSet h : st.handles) {

                // create new successor state
                State n_st = createState(st.index + 1, st.str, st.handles);

                // get best parent set given current handlers
                n_st.str[v] = getBestParentSet(v, h);

                // add new handlers
                for (int elim : h.set) {
                    n_st.handles.add(
                            new SIntSet(reduceAndIncreaseArray(h.set, v, elim)));
                }

                // update score
                n_st.updateSk();

                // add to candidates
                candidates.add(n_st);
            }

            if (exp_limit == 0 || st.index <= tw + exp_limit) {
                // put everything
                for (State c : candidates) {
                    addSuccessorState(c);
                }
            } else {
                // If we are over the treeshold, put only the best
                addSuccessorState(candidates.pollFirst());
            }

            solver.checkTime();

            // pf("%d - %d\n", open.size(), st.index);
        }

        if (completed) {
            return st.str;
        }

        return null;
    }

    ParentSet getBestParentSet(int v, SIntSet h) {
        // find best parent set for handler in this successor state
        for (ParentSet pSet : m_scores[v]) {
            // if parent set is available
            if (containsAll(pSet.parents, h.set)) {
                // set parent set
                return pSet;
            }
        }

        return null;
    }

    protected State createState(int i, ParentSet[] ps, TreeSet<SIntSet> handles) {
        return new State(i, ps, handles);
    }

    private void addSuccessorState(State n_st) {

        boolean toDropWorst = false;

        if (open.size() > queue_size) {

            if (n_st.f_sk < worstQueueScore) {
                // log.conclude("pruned");
                return;
            }

            toDropWorst = true;
        }

        // Drop worst element in queue, to make room!
        if (toDropWorst) {
            open.pollLast();
            worstQueueScore = open.last().f_sk;
        } else // If we didn't drop any element, check if we have to update the current
        // worst score!
        if (n_st.f_sk < worstQueueScore) {
            worstQueueScore = n_st.f_sk;
        }

        open.add(n_st);
    }

    @Override
    protected void clear() {
        super.clear();
        open = new TreeSet<State>();
    }

    protected void initHVector() {

        heuStr = new double[n_var];

        for (int i = tw + 1; i < n_var; i++) {
            int[] acceptable = new int[i];

            System.arraycopy(vars, 0, acceptable, 0, i);
            Arrays.sort(acceptable);
            int v = vars[i];

            for (ParentSet pSet : m_scores[v]) {
                // if every parent is preceding him in the order
                if (containsAll(pSet.parents, acceptable)) {
                    heuStr[v] = pSet.sk;
                    break;
                }
            }

        }
    }

    protected class State implements Comparable<State> {

        // Structure built so far
        public final ParentSet[] str;

        // Optimistic score map evaluation
        public double f_sk;

        // Near-maximal clique (size tw), for adding new variables
        final TreeSet<SIntSet> handles;

        // Next variable to explore (index in vars)
        protected final int index;

        public State(int index, ParentSet[] new_str, TreeSet<SIntSet> new_handles) {
            this.index = index;

            str = new ParentSet[n_var];
            cloneStr(new_str, str);

            handles = new TreeSet<SIntSet>();
            for (SIntSet s : new_handles) {
                handles.add(s);
            }
        }

        @Override
        public int compareTo(State other) {
            if (doubleEquals(f_sk, other.f_sk)) {
                if (index < other.index) {
                    return 1;
                } else {
                    return -1;
                }
            }

            if (f_sk < other.f_sk) {
                return 1;
            }
            return -1;
        }

        public void updateSk() {
            f_sk = 0;

            // Exact part
            for (int i = 0; i < index; i++) {
                int v = vars[i];

                f_sk += str[v].sk;
            }

            // Heuristic part
            for (int i = index; i < n_var; i++) {
                f_sk += heuStr[vars[i]];
            }

        }

        public String toString() {
            return f("index: %d, score: %.2f", index, f_sk);
        }
    }

}
