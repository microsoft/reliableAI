package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.data.SIntSet;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.wf;


/**
 * Sequential scorer
 */

public class SeqNewScorer extends SeqScorer {

    private static final Logger log = Logger.getLogger(
            SeqNewScorer.class.getName());

    public int new_cnt = 0;
    public int old_cnt = 0;
    public int both_cnt = 0;
    public int def_cnt = 0;

    Object lock2 = new Object();
    public Writer wr;

    @Override
    protected String getName() {
        return "New sequential selection";
    }

    public SeqNewScorer() {
        super();
    }

    @Override
    public void prepare() {
        super.prepare();

        def_cnt += n_var * (n_var - 1);
    }

    @Override
    public SeqNewSearcher getNewSearcher(int n) {
        return new SeqNewSearcher(n);
    }

    public class SeqNewSearcher extends SeqSearcher {

        private double[] w;

        public SeqNewSearcher(int n) {
            super(n);
        }

        @Override
        public void prepare() {
            super.prepare();

            computeW();
        }

        @Override
        public void checkBound(double sk, int[] n_pset) {

            boolean old_bound = false;
            boolean new_bound = false;

            for (int Z : n_pset) {

                boolean explore = true;

                int[] pset = reduceArray(n_pset, Z);

                double p = -pen(n, n_pset);
                double p_o = pen(n, pset);
                double p2 = (1 - dat.l_n_arity[Z]) * pen(n, pset);

                // OLD BOUND
                if (p + scores.get(new SIntSet(pset)) >= 0) {
                    old_bound = true;
                }

                // NEW BOUND
                if (w[Z] <= p2) {
                    new_bound = true;
                }
            }

            synchronized (lock2) {
                if (new_bound) {
                    new_cnt++;
                }

                if (old_bound) {
                    old_cnt++;
                }

                if (new_bound || old_bound) {
                    both_cnt++;
                }

                def_cnt++;

                try {
                    wf(wr, "%.2f %d %s\n", sk, n, Arrays.toString(n_pset));
                    wr.flush();
                } catch (IOException ex) {
                    p(ex.getMessage());
                }
            }

        }

        /*
         private void search() throws IOException {


         start = System.currentTimeMillis();
         elapsed = 0;

         int[] s = new int[0];

         exploreSubSet(s);
         }

         private void exploreSubSet(int[] s) throws IOException {

         if (!thereIsTime())
         return;

         double sk;

         synchronized (lock) {
         computed += 1;
         }

         //pf("%s\n", Arrays.toString(s));

         if (s.length > 0) {

         sk = score.computeScore(n, s);

         // p(Arrays.toString(s));

         if (sk > voidSk) {
         addScore(s, sk);
         }
         } else
         sk = voidSk;


         if (s.length >= max_pset_size)
         return;

         // Check for supersets to explore
         for (int Z = 0; Z < n_var && thereIsTime(); Z++) {

         if (Z <= n && Arrays.binarySearch(s, Z) >= 0)
         continue;

         boolean explore = true;

         int[] n_pset = expandArray(s, Z);
         double p = -pen(n, n_pset);
         double p_o = pen(n, s);
         double p2 = (1 - dat.l_n_arity[Z]) * pen(n, s);

         // OLD BOUND
         if (explore && oldBound) {
         if (p + sk > 0) {
         // if (p + (sk + p_o) > 0) {
         // wf(wr, "PRUNED2 %s \n", Arrays.toString(n_pset));
         explore = false;
         double wz = w[Z];
         boolean c = wz <= p2;
         }
         }

         // NEW BOUND
         if (explore && newBound) {
         if (w[Z] <= p2) {
         // wf(wr, "PRUNED %s \n", Arrays.toString(n_pset));
         explore = false;
         }
         }

         if (explore)
         exploreSubSet(n_pset);
         }
         }

         private void expandParentSet(int[] s) throws IOException {

         /*
         // Find the best Y with highest w(X, Y) + Pen(X, s + Y)
         int best_Y = -1;
         double best_S = -Double.MAX_VALUE;

         // Pen(X, s)
         double pen = pen(n, s);

         for (int Y = 0; Y < n_var; Y++) {
         if (Y == n)
         continue;

         // w(X, Y) + Pen(X, s) * |Y|
         double S = w[Y] + pen*dat.l_n_arity[Y];
         if (S > best_S) {
         best_Y = Y;
         best_S = S;
         }

         }
         // System.out.println(Arrays.toString(set));


         }
         */

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
                double n_w = mi.computeMi(i, n);

                n_w -= Math.max(ll.computeLL(i), ll.computeLL(n));
            }
        }

        private double pen(int n, int[] pset) {
            double pe = -Math.log(dat.n_datapoints) * (dat.l_n_arity[n] - 1) / 2;

            for (int p : pset) {
                pe *= dat.l_n_arity[p];
            }
            return pe;
        }

    }

    /* public static boolean incrementPset(int[] s, int i, int n_var) {

     if (i < 0) {
     return false;
     }

     // Try to increment set at position thread
     s[i]++;

     // Check if we have to backtrack
     if (s[i] > (n_var - (s.length - i))) {
     boolean cnt = incrementPset(s, i - 1, n_var);

     if (cnt) {
     s[i] = s[i - 1] + 1;
     }
     return cnt;
     }

     return true;
     } */
}

