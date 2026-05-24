package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.analyze.Entropy;
import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.learn.scorer.concurrency.NotifyingThread;
import ch.idsia.blip.core.learn.scorer.concurrency.ThreadCompleteListener;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.data.map.ArrayHashingStrategy;
import ch.idsia.blip.core.utils.data.map.TCustomHashMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static ch.idsia.blip.core.learn.scorer.SeqScorer.incrementPset;
import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.pf;
import static java.lang.StrictMath.max;


/**
 * Sequential scorer, ULTIMATEEEE
 */

public class SeqUltScorer extends BaseScorer {

    private List<Map<int[], Double>> t_sc;
    private List<Map<int[], Double>> t_h;

    private static final Logger log = Logger.getLogger(
            SeqUltScorer.class.getName());

    private Entropy ent;

    private BIC bic;

    public int prune_x;
    public int prune_y;
    public int prune_xy;

    public int prune_xpi;
    public int prune_ypi;
    public int prune_xypi;

    private double[] voidSk;
    private double[] voidH;
    public int prune_new;

    @Override
    protected String getName() {
        return "Sequential Ultimate selection";
    }

    public void searchAll() throws InterruptedException, IOException {

        t_sc = new ArrayList<Map<int[], Double>>();
        t_h = new ArrayList<Map<int[], Double>>();
        for (int i = 0; i < n_var; i++) {
            t_sc.add(
                    new TCustomHashMap<int[], Double>(new ArrayHashingStrategy()));
            t_h.add(
                    new TCustomHashMap<int[], Double>(new ArrayHashingStrategy()));
        }

        voidSk = new double[n_var];
        voidH = new double[n_var];

        ent = new Entropy(dat);
        bic = new BIC(dat);

        if (verbose > 0) {
            pf("Executing with: \n");
            pf("%-12s: %s \n", "code", this.getClass().getName());
            pf("%-12s: %d \n", "threads", thread_pool_size);
            pf("%-12s: %.2f \n", "max_time", max_exec_time);
            pf("%-12s: %d \n", "max_degree", max_pset_size);
        }

        for (int n = 0; n < n_var; n++) {
            double sk = score.computeScore(n);

            t_sc.get(n).put(new int[0], sk);
            voidSk[n] = sk;

            double h = ent.computeH(n);

            t_h.get(n).put(new int[0], h);
            voidH[n] = h;
        }

        // For each s size, search all the variables, record the joint entropies
        for (int i = 1; i <= max_pset_size; i++) {
            if (verbose > 1) {
                safeLogf("\nNew level: %d - %.2f\n", i, (System.currentTimeMillis() - start) / 1000.0 );
            }
            Thread t1 = new Thread(
                    new UltExecutor(thread_pool_size, 0, n_var, this, i));

            t1.start();
            t1.join();
        }

        if (verbose > 1) {
            safeLogf("\nDone! \n");
        }

        // Write scores anyway TODO
        scoreWriter = new ScoreWriter(this, ph_scores, 0, n_var, 0);
        Thread t2 = new Thread(scoreWriter);

        t2.start();
        for (int i = 0; i < n_var; i++) {
            scoreWriter.add(i, t_sc.get(i));
        }

        t2.join();

    }

    @Override
    public BaseSearcher getNewSearcher(int n) {
        return null;
    }

    public class SeqUltSearcher extends BaseSearcher {

        Map<int[], Double> l_sc;
        Map<int[], Double> l_h;

        private final int pset_size;

        SeqUltSearcher(int n, int pset_size) {
            super(n);
            this.pset_size = pset_size;

            l_sc = new TCustomHashMap<int[], Double>(new ArrayHashingStrategy());
            l_h = new TCustomHashMap<int[], Double>(new ArrayHashingStrategy());
        }

        /**
         * Evaluate the parent sets of the variable in the available time, following an heuristic ordering.
         */
        @Override
        public void run() {

            first();

            int[] pset = new int[pset_size];

            for (int i = 0; i < pset_size; i++) {
                pset[i] = i;
            }

            boolean cnt = true;

            while (cnt) {
                evaluate(pset);

                cnt = nextPset(pset);
            }

            if (verbose > 1) {
                safeLogf("# %d - %.2f\n", n, (System.currentTimeMillis() - start) / 1000.0);
            }

            aggregate(n, l_sc, l_h);
        }

        private void first() {
            MutualInformation mi = new MutualInformation(dat);
            LogLikelihood ll = new LogLikelihood(dat);
            double ll_n = ll.computeLL(n);

            double[] w = new double[n_var];

            for (int i = 0; i < n_var; i++) {
                if (i == n) {
                    continue;
                }

                // double ll_i = ll.computeLL(n, i);

                double m = mi.computeMi(n, i) * dat.n_datapoints;
                double l = max(ll_n, ll.computeLL(i));

                // double l = - (ll_n -ll.go(n, thread)) / dat.n_datapoints;

                w[i] = m - l;
            }

        }

        private boolean nextPset(int[] pset) {
            return incrementPset(pset, pset.length - 1, dat.n_var);
        }

        private void evaluate(int[] pset) {
            if (Arrays.binarySearch(pset, n) >= 0) {
                return;
            }

            SIntSet s = new SIntSet(cloneArray(pset));

            /*
             for (int Z: s) {
             int[] o_pset = reduceArray(s, Z);
             double p = -pen(n, s);
             if (o_pset.length > 1) {
             double o_sk = t_sc.get(n).get(new SIntSet(o_pset));
             if(p + o_sk >= 0){
             p("old_bound");
             }
             }
             }*/

            /* p(Arrays.toString(s));
             if (n == 1 && Arrays.toString(s).equals("[0, 2, 6, 7]"))
             p("bro");*/

            // Pruning
            boolean toPruneXPi = (pruneXPi(pset, pset));

            boolean toPruneYPi = (pruneYPi(pset, pset));

            boolean toPruneX = (pruneX(pset, pset));

            boolean toPruneY = (pruneY(pset, pset));

            /*
             if (!toPruneXPi && toPruneYPi)
             p("cia"); */

            synchronized (lock) {
                if (toPruneXPi) {
                    prune_xpi++;
                }

                if (toPruneYPi) {
                    prune_ypi++;
                }

                if (toPruneXPi || toPruneYPi) {
                    prune_xypi++;
                }

                if (toPruneX) {
                    prune_x++;
                }

                if (toPruneY) {
                    prune_y++;
                }

                if (toPruneX || toPruneY) {
                    prune_xy++;
                }

                if (toPruneXPi || toPruneY) {
                    prune_new++;
                }
            }

            if (toPruneXPi || toPruneYPi) {
                 // return;
            }

            double sk = bic.computeScore(n, pset);

            l_sc.put(s.set, sk);

            double h = ent.computeHCond(n, pset);

            l_h.put(s.set, h);

            // pf("%d - %s\n", n, Arrays.toString(pset));
        }

        /*
        private double pen(int n, int[] pset) {
            double pe = -Math.log(dat.n_datapoints) * (dat.l_n_arity[n] - 1) / 2;

            for (int p : pset) {
                pe *= dat.l_n_arity[p];
            }
            return pe;
        } */

        private boolean pruneY(int[] orig, int[] pset) {

            SIntSet s = new SIntSet(pset);

            for (int y : orig) {
                if (find(y, pset)) {
                    continue;
                }

                double pen = bic.getPenalization(n, pset);

                // N*H(Y|\Pi^') <= (1 - |Y|) Pen(X|\Pi^*)
                if (check(pen, s, y, y)) {
                    return true;
                }
            }

            if (pset.length >= 1) {
                for (int p : pset) {
                    if (pruneY(pset, reduceArray(pset, p))) {
                        return true;
                    }
                }
            }

            return false;
        }

        private boolean pruneX(int[] orig, int[] pset) {

            SIntSet s = new SIntSet(pset);

            for (int y : orig) {
                if (find(y, pset)) {
                    continue;
                }

                double pen = bic.getPenalization(n, pset);

                // N*H(X|\Pi^') <= (1 - |Y|) Pen(X|\Pi^*)
                if (check(pen, s, n, y)) {
                    return true;
                }
            }

            if (pset.length >= 1) {
                for (int p : pset) {
                    if (pruneX(pset, reduceArray(pset, p))) {
                        return true;
                    }
                }
            }

            return false;
        }

        private boolean pruneYPi(int[] orig, int[] pset) {

            SIntSet s = new SIntSet(pset);

            for (int y : orig) {
                if (find(y, pset)) {
                    continue;
                }

                double pen = bic.getPenalization(n, pset);

                // N*H(Y|\Pi^') <= (1 - |Y|) Pen(X|\Pi^*)
                if (checkPi(pen, s, y, y)) {
                    return true;
                }
            }

            if (pset.length >= 1) {
                for (int p : pset) {
                    if (pruneYPi(pset, reduceArray(pset, p))) {
                        return true;
                    }
                }
            }

            return false;
        }

        private boolean pruneXPi(int[] orig, int[] pset) {

            SIntSet s = new SIntSet(pset);

            for (int y : orig) {
                if (find(y, pset)) {
                    continue;
                }

                double pen = bic.getPenalization(n, pset);

                // N*H(X|\Pi^') <= (1 - |Y|) Pen(X|\Pi^*)
                if (checkPi(pen, s, n, y)) {
                    return true;
                }
            }

            if (pset.length >= 1) {
                for (int p : pset) {
                    if (pruneXPi(pset, reduceArray(pset, p))) {
                        return true;
                    }
                }
            }

            return false;
        }

        private boolean checkPi(double pen, SIntSet s, int v, int y) {
            Double h = t_h.get(v).get(s.set);

            if (h == null) {
                return false;
            }

            double t1 = dat.n_datapoints * h;
            double t2 = (dat.l_n_arity[y] - 1) * pen;

            // if (t2 > t1)
            // p("ciao");

            // if (s.set.length >=3)
            // p("CIA");
            // if (t1 <=t2) {
            // p(ent.computeHCond(v, s.set));
            // p("ca");
            // }

            return t1 <= t2;
        }

        private boolean check(double pen, SIntSet s, int v, int y) {
            Double h = t_h.get(v).get(s.set);

            if (h == null) {
                return false;
            }
           //  double sc2 = bic.computeScore(n, s.set);
           //  double pen2 = bic.getPenalization(n, s.set);
            // double t = sc2 + pen2;

            h = voidH[v];

            double t1 = dat.n_datapoints * h;
            double t2 = (dat.l_n_arity[y] - 1) * pen;

            // if (t2 > t1)
            // p("ciao");

            // if (s.set.length >=3)
            // p("CIA");

            return t1 <= t2;
        }
    }

    private void aggregate(int n, Map<int[], Double> l_sc, Map<int[], Double> l_h) {
        synchronized (lock) {
            t_sc.get(n).putAll(l_sc);
            t_h.get(n).putAll(l_h);
        }
    }

    private class UltExecutor implements Runnable, ThreadCompleteListener {

        private final Object lock = new Object();
        private final int max_thread;
        private final SeqUltScorer scorer;
        private final int start;
        private final int end;
        private final int pset_size;
        private Integer crt_thread;
        private int completed;

        UltExecutor(int max_thread, int start, int end, SeqUltScorer scorer, int pset_size) {
            this.max_thread = max_thread;
            this.scorer = scorer;
            this.start = start;
            this.end = end;
            this.pset_size = pset_size;
        }

        @Override
        public void run() {

            completed = 0;

            crt_thread = 0;

            int v = start;

            while (v < end) {

                try {
                    synchronized (lock) {
                        if (crt_thread < max_thread) {
                            crt_thread++;
                            Runnable r = scorer.getNewSearcher(v, pset_size);
                            NotifyingThread t = new NotifyingThread(r);

                            t.addListener(this);
                            t.start();
                            v++;
                        } else {
                            lock.wait();
                        }
                    }

                } catch (InterruptedException e) {
                    RandomStuff.logExp(log, e);
                }
            }

            while (completed != (end - start)) {
                synchronized (lock) {
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        RandomStuff.logExp(log, e);
                    }
                }
            }
        }

        @Override
        public void notifyOfThreadComplete() {
            synchronized (lock) {
                crt_thread--;
                completed++;
                lock.notify();
            }
        }
    }

    private Runnable getNewSearcher(int v, int pset_size) {
        return new SeqUltSearcher(v, pset_size);
    }
}

