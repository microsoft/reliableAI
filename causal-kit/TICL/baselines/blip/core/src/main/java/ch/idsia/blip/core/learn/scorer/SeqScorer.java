package ch.idsia.blip.core.learn.scorer;


import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;


/**
 * Sequential scorer
 */

public class SeqScorer extends BaseScorer {

    private static final Logger log = Logger.getLogger(SeqScorer.class.getName());

    @Override
    protected String getName() {
        return "Sequential selection";
    }

    public SeqScorer() {
        super();
    }

    public static boolean incrementPset(int[] pset, int i, int n_var) {

        if (i < 0) {
            return false;
        }

        // Try to increment set at position thread
        pset[i]++;

        // Check if we have to backtrack
        if (pset[i] > (n_var - (pset.length - i))) {
            boolean cnt = incrementPset(pset, i - 1, n_var);

            if (cnt) {
                pset[i] = pset[i - 1] + 1;
            }
            return cnt;
        }

        return true;
    }

    @Override
    public SeqSearcher getNewSearcher(int n) {
        return new SeqSearcher(n);
    }

    public class SeqSearcher extends BaseSearcher {

        public SeqSearcher(int n) {
            super(n);
        }

        /**
         * Evaluate the parent sets of the variable in the available time, following an heuristic ordering.
         */
        @Override
        public void run() {

            if (verbose > 2) {
                log.info(
                        String.format("Starting with: %d, max time: %.2f", n,
                        max_exec_time));
            }

            prepare();

            int pset_size = 2;

            while (thereIsTime()) {

                int[] pset = new int[pset_size];

                for (int i = 0; i < pset_size; i++) {
                    pset[i] = i;
                }

                // System.out.printf("%d - %d\n", dat.n_var, n);

                boolean cnt = true;

                while (cnt && thereIsTime()) {

                    // System.out.println(Arrays.toString(set));

                    if (Arrays.binarySearch(pset, n) < 0) {

                        double sk = score.computeScore(n, pset);

                        addScore(cloneArray(pset), sk);

                        checkBound(sk, pset);
                    }

                    cnt = incrementPset(pset, pset.length - 1, dat.n_var);

                }

                if (max_pset_size > 0 && pset_size >= max_pset_size) {
                    break;
                }

                pset_size++;

            }

            if (verbose > 2) {
                logf("ending with: %d, elapsed: %.2f, num evaluated %d", n,
                        m_elapsed, score.numEvaluated);
            }

            conclude();

        }

        protected void checkBound(double sk, int[] pset) {}

        /*
         protected void checkBound(double sk, int[][] p_values, int[] s) {

         // de Campos and Ji style
         int k_i = 0;

         for (int[] p_v : p_values) {
         if (p_v.length > 0) {
         k_i++;
         }
         }

         double bound = (-Math.log(dat.l_n_arity[n]) * k_i);

         if (sk > bound) {
         pf("%s Ji and Campos strikes again! \n", Arrays.toString(s));
         }
         }
         */
    }

}

