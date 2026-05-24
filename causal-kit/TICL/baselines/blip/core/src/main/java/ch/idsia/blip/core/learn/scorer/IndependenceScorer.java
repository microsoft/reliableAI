package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.learn.scorer.utils.OpenParentSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.map.ArrayHashingStrategy;
import ch.idsia.blip.core.utils.data.map.TCustomHashSet;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.wf;


/**
 * Independence scorer
 * <p/>
 * Questa sara la mia salvezza. O la mia dannazione.
 */

public class IndependenceScorer extends BaseScorer {

    private static final Logger log = Logger.getLogger(
            IndependenceScorer.class.getName());

    // Maximum size for queue
    public long queue_size;

    public final long max_queue_size = (long) Math.pow(2, 20);

    // public final long max_cache_size = (long) Math.pow(2, 10);

    // Maximum size for cache
    // public long cache_size;

    @Override
    protected String getName() {
        return "Independence selection";
    }

    public IndependenceScorer() {
        super();
    }

    /**
     * @param n_var      number of variables available as parent
     * @param max_parent max number of parents
     * @return estimated total number of possible parent set
     */
    public static double totNumParentSet(int n_var, int max_parent) {
        double n = 0;

        for (int i = 2; i <= max_parent; i++) {

            double k = 1;

            for (int j = n_var - 1; j > ((n_var - 1) - i); j--) {
                k *= j;
            }
            for (int j = 1; j <= i; j++) {
                k /= j;
            }

            n += k;
        }
        return n;
    }

    @Override
    protected void preamble(BaseScorer sc, Writer wr) throws IOException {
        super.preamble(sc, wr);
        wf(wr, "# Time for variable: %.2f \n", max_searcher_time);
    }

    @Override
    public void prepare() {
        super.prepare();

        queue_size = Math.min((long) Math.pow(dat.n_var, 3), max_queue_size);
        if (verbose > 1) {
            logf("cache size: %d \n", queue_size);
        }

        // cache_size = Math.min((long) Math.pow(dat.n_var, 2), max_cache_size);
        // logf(1, "queue size: %d \n", queue_size);
    }

    @Override
    public IndependenceSearcher getNewSearcher(int n) {
        return new IndependenceSearcher(n);
    }

    public class IndependenceSearcher extends BaseSearcher {

        /**
         * Holder of the currently worst score saved in queue for evaluation
         */
        private double worstQueueScore;

        /**
         * r
         * Queue for parent set to examine
         */
        private TreeSet<OpenParentSet> open;

        /**
         * Set of already considered parent sets
         */
        private Set<int[]> closed;

        private int[] new_pset;

        public IndependenceSearcher(int in_n) {
            super(in_n);
        }

        @Override
        protected void prepare() {
            super.prepare();

            // Initialize everything
            closed = new TCustomHashSet<int[]>(new ArrayHashingStrategy()); // Parent set already seen
            open = new TreeSet<OpenParentSet>(); // Parent set to evaluate

            if (max_pset_size <= 1) {
                return;
            }

            // Consider all two-parent set for evaluation
            for (int i = 0; i < parents.length; i++) {
                for (int j = i + 1; j < parents.length; j++) {

                    addParentSetToEvaluate(new int[] { parents[i]}, parents[j],
                            null);
                }
            }

        }

        /**
         * Evaluate the parent sets of the variable in the available time, following an heuristic ordering.
         */
        @Override
        public void run() {

            if (verbose > 2) {
                logf("Starting with: %d, max time: %.2f", n, max_exec_time);
            }

            prepare();

            // Consider all the parent set for evaluation!
            while (!open.isEmpty() && thereIsTime()) {

                OpenParentSet p = open.pollFirst();

                if (p == null) {
                    continue;
                }


                // Check if all the subset have already been evaluated; else re-put into the list
                evalutateSubsets(p.s);

                evaulateParentSet(p.s, p.p_values, p.new_p);
            }

            if (verbose > 2) {
                logf("ending with: %d, elapsed: %.2f, num evaluated %d", n,
                        m_elapsed, score.numEvaluated);
            }

            conclude();
        }

        private void evalutateSubsets(int[] s) {

            if (s.length <= 2) {
                return;
            }

            for (int p : s) {
                int[] set_new = reduceArray(s, p);

                boolean explored = scores.containsKey(set_new);

                if (!explored && thereIsTime()) {
                    evalutateSubsets(set_new);

                    evaulateParentSet(set_new, null, -1);
                }
            }

        }

        /**
         * Compute the score for a given parent set.
         * If we believe they will be useful, add the children parent set to evaluation in the queue.
         */
        void evaulateParentSet(int[] s, int[][] p_values, int new_p) {

            if (scores.containsKey(s)) {
                return;
            }

            if (p_values == null) {
                p_values = score.computeParentSetValues(s);
            } else {
                p_values = score.expandParentSetValues(s, p_values, new_p);
            }

            // Evaluate parent set
            double sk = getScore(s, p_values);

            addScore(s, sk);

            // Decide if we add his children
            if (max_pset_size > 0 && s.length >= max_pset_size) {
                return;
            }

            /*
             // de Campos and Ji style
             int k_i = 0;

             for (int[] p_v : p_values) {
             if (p_values.length > 0) {
             k_i++;
             }
             }

             if (sk > (-Math.log(dat.l_n_arity[n]) * k_i)) {
             // log.conclude("Ji and Campos strikes again!");
             return;
             }*/

            // Consider children parent set for evaluation
            for (int p3 : parents) {

                if (!thereIsTime()) {
                    return;
                }

                if (Arrays.binarySearch(s, p3) >= 0) {
                    continue;
                }

                addParentSetToEvaluate(s, p3, p_values);
            }
        }

        protected double getScore(int[] set_p, int[][] p_values) {
            return score.computeScore(n, set_p, p_values);
        }

        /**
         * Add a new parent set to the list of the ones to evaluate
         *
         * @param p1 original parent set
         * @param p2 new parent
         */
        void addParentSetToEvaluate(int[] p1, int p2, int[][] p_values) {

            // Estimate score for new parent set
            double mean_sk = score.computePrediction(n, p1, p2, scores);

            boolean toDropWorst = false;

            if (open.size() > queue_size) {

                if (mean_sk < worstQueueScore) {
                    // log.conclude("pruned");
                    return;
                }

                toDropWorst = true;
            }

            // Compute new set hash
            int[] new_pset = ArrayUtils.expandArray(p1, p2);

            // Check if it hasn't been already evaluated
            if (closed.contains(new_pset)) {
                return;
            }

            closed.add(new_pset);

            // Drop worst element in queue, to make room!
            if (toDropWorst) {
                open.pollLast();
                worstQueueScore = open.last().sk;
            } else // If we didn't drop any element, check if we have to update the current
            // worst score!
            if (mean_sk < worstQueueScore) {
                worstQueueScore = mean_sk;
            }

            open.add(new OpenParentSet(new_pset, p2, mean_sk, p_values));
        }

    }

}

