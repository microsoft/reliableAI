package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.utils.analyze.MutualInformation;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.learn.scorer.SeqScorer.incrementPset;
import static java.lang.StrictMath.max;


/**
 * Sequential scorer
 */

public class SeqAdvScorer extends BaseScorer {

    private static final Logger log = Logger.getLogger(
            SeqAdvScorer.class.getName());

    @Override
    protected String getName() {
        return "Sequential Advanced selection";
    }

    public SeqAdvScorer() {
        super();
    }

    @Override
    public SeqAdvSearcher getNewSearcher(int n) {
        return new SeqAdvSearcher(n);
    }

    public class SeqAdvSearcher extends BaseSearcher {

        public SeqAdvSearcher(int n) {
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

            ThreadMXBean bean = ManagementFactory.getThreadMXBean();
            double start = bean.getCurrentThreadCpuTime();
            double elapsed = 0;

            int pset_size = 2;

            MutualInformation mi = new MutualInformation(dat);
            LogLikelihood ll = new LogLikelihood(dat);
            double ll_n = ll.computeLL(n);

            double[] w = new double[n_var];

            for (int i = 0; i < n_var; i++) {
                if (i == n) {
                    continue;
                }

                double ll_i = ll.computeLL(n, i);

                double m = mi.computeMi(n, i) * dat.n_datapoints;
                double l = max(ll_n, ll.computeLL(i));

                // double l = - (ll_n -ll.go(n, thread)) / dat.n_datapoints;

                w[i] = m - l;
            }

            while (elapsed < max_exec_time) {

                int[] pset = new int[pset_size];

                for (int i = 0; i < pset_size; i++) {
                    pset[i] = i;
                }

                // System.out.printf("%d - %d\n", dat.n_var, n);

                boolean cnt = true;

                while (cnt && (elapsed < max_exec_time)) {

                    // System.out.println(Arrays.toString(set));

                    if (Arrays.binarySearch(pset, n) < 0) {

                        double sk = score.computeScore(n, pset);

                        double bestScore;

                        bestScore = voidSk;

                        if (sk > bestScore) {
                            addScore(pset, sk);
                        }
                    }

                    cnt = incrementPset(pset, pset.length - 1, dat.n_var);

                    elapsed = (bean.getCurrentThreadCpuTime() - start)
                            / 1000000000;
                }

                pset_size++;

                if (max_pset_size > 0 && pset_size >= max_pset_size) {
                    break;
                }

            }

            if (verbose > 2) {
                log.info(
                        String.format(
                                "ending with: %d, elapsed: %.2f, num evaluated %d",
                                n, elapsed, score.numEvaluated));
            }

            conclude();

        }

    }

}

