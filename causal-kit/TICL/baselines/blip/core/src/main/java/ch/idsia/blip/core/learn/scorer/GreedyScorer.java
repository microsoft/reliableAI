package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.map.ArrayHashingStrategy;
import ch.idsia.blip.core.utils.data.map.TCustomHashSet;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;


public class GreedyScorer extends BaseScorer {

    private static final Logger log = Logger.getLogger(
            GreedyScorer.class.getName());

    /**
     * Queue size limit (for memory!)
     */
    private final long max_queue_size = (long) Math.pow(2, 20);

    /**
     * Maximum size for queue
     */
    private long queue_size;

    @Override
    protected String getName() {
        return "Greedy scoring";
    }

    public GreedyScorer() {
        super();
    }

    @Override
    public void prepare() {
        super.prepare();

        queue_size = (long) Math.pow(dat.n_var, 3);
        if (queue_size > max_queue_size) {
            queue_size = max_queue_size;
        }
    }

    @Override
    public GreedySearcher getNewSearcher(int n) {
        return new GreedySearcher(n);
    }

    /**
     * Entry of a parent set in the linked-list queue.
     */
    public static class ParentSetEntry implements Comparable<ParentSetEntry> {

        public final int[] s;

        public final double sk;

        public final int[][] p_values;

        /**
         * Default constructor.
         *
         * @param pset hash of the parent set base
         * @param sk   score of the parent set
         */
        ParentSetEntry(int[] pset, double sk, int[][] p_values) {
            this.s = pset;
            this.sk = sk;
            this.p_values = p_values;
        }

        @Override
        public int compareTo(ParentSetEntry other) {
            if (sk > other.sk) {
                return 1;
            }
            return -1;
        }

        public String toString() {
            return String.format("(%s %.3f)", Arrays.toString(s), sk);
        }

    }


    private class GreedySearcher extends BaseSearcher {

        /**
         * Queue for parent set to examine
         */
        private TreeSet<ParentSetEntry> open;

        /**
         * Set of already considered parent sets
         */
        private Set<int[]> closed;

        /**
         * Holder of the currently worst score saved in queue for evaluation
         */
        private double worstQueueScore;

        GreedySearcher(int in_n) {
            super(in_n);
        }

        /**
         * Evaluate the parent sets of the variable in the available time, following an heuristic ordering.
         */
        @Override
        public void run() {

            ThreadMXBean bean = ManagementFactory.getThreadMXBean();
            double start = bean.getCurrentThreadCpuTime();
            double elapsed = 0;

            prepare();

            if (verbose > 2) {
                log.info(
                        String.format("Starting with: %d, max time: %.2f", n,
                        max_exec_time));
            }

            // int arity = dat.l_n_arity[n];

            // Initialize everything
            closed = new TCustomHashSet<int[]>(new ArrayHashingStrategy()); // Parent set already seen
            open = new TreeSet<ParentSetEntry>(); // Parent set to evaluate

            // Compute one-scores
            for (int p = 0; p < dat.n_var; p++) {

                if (p == n) {
                    continue;
                }

                double sk;

                sk = oneScores.get(p);

                addScore(p, sk);

                addParentSetToEvaluate(new int[] { p}, sk, null);
            }

            if (max_exec_time == 0) {
                max_exec_time = Integer.MAX_VALUE;
            }

            // Consider all the parent set for evaluation!
            while (!open.isEmpty() && (elapsed < max_exec_time)) {

                ParentSetEntry pset = open.pollLast();

                if (pset == null) {
                    continue;
                }

                for (int p2 = 0; (p2 < dat.n_var) && (elapsed < max_exec_time); p2++) {

                    if (p2 == n) {
                        continue;
                    }

                    evaluteParentSet(n, pset.s, p2, pset.p_values);

                    elapsed = (bean.getCurrentThreadCpuTime() - start)
                            / 1000000000;
                }
            }

            if (verbose > 2) {
                log.info(
                        String.format(
                                "ending with: %d, elapsed: %.2f, num evaluated %d",
                                n, elapsed, score.numEvaluated));
            }

            // synchronized (scorer) {
            if (verbose > 0) {
                System.out.println("... finishing " + n);
            }

            conclude();
        }

        private void evaluteParentSet(int n, int[] old, int p2, int[][] p_values) {

            if (Arrays.binarySearch(old, p2) >= 0) {
                return;
            }

            int[] pars = ArrayUtils.expandArray(old, p2);

            if (max_pset_size > 0 && pars.length >= max_pset_size) {
                return;
            }

            if (closed.contains(pars)) {
                return;
            }

            closed.add(pars);

            if (p_values == null) {
                p_values = score.computeParentSetValues(pars);
            } else {
                p_values = score.expandParentSetValues(pars, p_values, p2);
            }

            double sk = score.computeScore(n, pars, p_values);

            // System.out.println(Arrays.toString(pars));

            if ((sk > voidSk)) {

                addScore(pars, sk);

                addParentSetToEvaluate(pars, sk, p_values);
            }

        }

        private void addParentSetToEvaluate(int[] p, double sk, int[][] p_values) {

            boolean toDropWorst = false;

            if (open.size() > queue_size) {

                if (sk < worstQueueScore) {
                    // log.conclude("pruned");
                    return;
                }

                toDropWorst = true;
            }

            // Drop worst element in queue, to make room!
            if (toDropWorst) {
                open.pollLast();
                worstQueueScore = open.last().sk;
            } else // If we didn't drop any element, check if we have to update the current
            // worst score!
            if (sk < worstQueueScore) {
                worstQueueScore = sk;
            }

            open.add(new ParentSetEntry(p, sk, p_values));
        }
    }
}
