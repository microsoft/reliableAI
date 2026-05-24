package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.hash.TIntDoubleHashMap;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.pf;
import static ch.idsia.blip.core.utils.RandomStuff.wf;
import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;


public class IndependenceNewScorer extends IndependenceScorer {

    public Writer wr;
    public Writer wr2;

    public TIntDoubleHashMap[] oneScoressss;

    @Override
    public void prepare() {
        super.prepare();

        oneScoressss = new TIntDoubleHashMap[dat.n_var];
    }

    @Override
    public IndependenceSearcher getNewSearcher(int n) {
        return new IndependenceNewSearcher(n);
    }

    public class IndependenceNewSearcher extends IndependenceSearcher {

        private double[] w;

        Object lock = new Object();

        private int cnt;

        private int cnt_old;
        private double cnt_old_t;

        private int cnt_new;
        private double cnt_new_t;

        public IndependenceNewSearcher(int n) {
            super(n);
        }

        @Override
        protected void prepare() {
            super.prepare();
            computeW();

            cnt = 0;
        }

        private void computeW() {
            MutualInformation mi = new MutualInformation(dat);
            LogLikelihood ll = new LogLikelihood(dat);

            w = new double[n_var];
            for (int i = 0; i < n_var; i++) {
                if (i == n) {
                    continue;
                }

                w[i] = mi.computeMi(i, n)
                        - Math.max(ll.computeLL(i), ll.computeLL(n));
                // double n_w = mi.computeMi(i, n);
                // n_w -= Math.max(ll.go(i), ll.go(n));
            }
        }

        private double pen(int n, int[] pset) {
            double pe = -Math.log(dat.n_datapoints) * (dat.l_n_arity[n] - 1) / 2;

            for (int p : pset) {
                pe *= dat.l_n_arity[p];
            }
            return pe;
        }

        @Override
        protected void conclude() {
            super.conclude();
            if (cnt_old > 0 && cnt_new > 0) {
                pf("WOAH %.2f %.2f \n", cnt_old_t / cnt_old, cnt_new_t / cnt_new);
            }

            oneScoressss[n] = oneScores;
        }

        @Override
        protected void addScore(int[] n_pset, double sk) {
            super.addScore(n_pset, sk);

            // check if there is pa better subset

            boolean pruned = false;

            int type = 1;

            for (int Z : n_pset) {

                int[] pset = reduceArray(n_pset, Z);

                double p = -pen(n, n_pset);
                double p_o = pen(n, pset);
                double p2 = (1 - dat.l_n_arity[Z]) * pen(n, pset);
                double bic_o = scores.get(new SIntSet(pset));
                double ll_o = bic_o + p_o;

                type = 1;

                // OLD BOUND
                if (p + bic_o >= 0) {
                    type += 1;
                    if (k(ll_o)) {
                        cnt_old_t += ll_o;
                        cnt_old++;
                    }
                }

                // NEW BOUND
                if (w[Z] <= p2) {
                    type += 2;
                    if (k(ll_o)) {
                        cnt_new_t += ll_o;
                        cnt_new++;
                    }
                }

                // if (type == 3) {
                // double v = scores.get(new SIntSet(s));
                // if (v < -2000) {
                // p(scores.get(new SIntSet(s)));
                // p(score.computeScore(n, s));
                // }
                // }


                if (type > 1) {
                    break;
                }
            }

            // if (checkToPrune(sk, n_pset, scores))
            // type=5;

            if (sk == -Double.MAX_VALUE) {
                return;
            }

            synchronized (lock) {
                cnt++;

                try {
                    wf(wr, "%d %.2f %d  %d %s\n", cnt, sk, type, n,
                            Arrays.toString(n_pset));
                    wr.flush();

                    if (!checkToPrune(sk, n_pset, scores)) {
                        wf(wr2, "%d %.2f %d  %d %s\n", cnt, sk, type, n,
                                Arrays.toString(n_pset));
                    }
                    wr2.flush();
                } catch (IOException ex) {
                    p(ex.getMessage());
                }

            }
        }

        private boolean k(double l) {
            if (Double.isNaN(l)) {
                return false;
            }
            if (Double.isInfinite(l) || Double.isInfinite(-l)) {
                return false;
            }
            if (Double.MAX_VALUE == l || Double.MAX_VALUE == -l) {
                return false;
            }
            return true;
        }
    }
}
