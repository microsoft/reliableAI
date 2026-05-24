package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.scorer.concurrency.NotifyingThread;
import ch.idsia.blip.core.learn.scorer.concurrency.ThreadCompleteListener;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getBayesianNetwork;


public class LLEvalHidden extends App implements ThreadCompleteListener {

    private static final Logger log = Logger.getLogger(
            LLEvalHidden.class.getName());

    private int crt_thread;

    private double ll;

    private int done;

    private VariableElimination vEl;

    private BayesianNetwork bn;

    private int[] ord;

    public LLEvalHidden(int threads) {
        this.thread_pool_size = threads;
        init();
    }

    public static double ex(String dat_path, String bn_path) throws FileNotFoundException {
        return ex(dat_path, bn_path, 0);
    }

    public static double ex(String dat_path, String bn_path, int threads) throws FileNotFoundException {
        LLEvalHidden l = new LLEvalHidden(threads);

        return l.go(dat_path, bn_path);
    }

    public double go(String dat_path, String bn_path) throws FileNotFoundException {

        DatFileLineReader ds = new DatFileLineReader(dat_path);

        ds.readMetaData();

        ll = 0;
        done = 0;

        bn = getBayesianNetwork(bn_path);
        ord = bn.getTopologicalOrder();
        vEl = new VariableElimination(bn, false);

        int n_datapoints = 0;

        while (!ds.concluded) {
            short[] sample = ds.next();

            try {
                synchronized (lock) {
                    if (crt_thread < thread_pool_size * 10) {
                        crt_thread++;
                        Runnable r = getNewSearcher(sample);
                        NotifyingThread t = new NotifyingThread(r);

                        t.addListener(this);
                        t.start();

                        n_datapoints++;
                    } else {
                        lock.wait();
                    }
                }

                while (done != n_datapoints) {
                    synchronized (lock) {
                        lock.wait();
                    }
                }

            } catch (InterruptedException e) {
                RandomStuff.logExp(log, e);
            }
        }

        return ll / done;
    }

    private Runnable getNewSearcher(short[] sample) {
        return new HiddenSearcher(sample);
    }

    @Override
    public void notifyOfThreadComplete() {
        synchronized (lock) {
            crt_thread--;
            lock.notify();
        }
    }

    private class HiddenSearcher implements Runnable {

        private final short[] sample;

        public HiddenSearcher(short[] sample) {
            this.sample = sample;
        }

        @Override
        public void run() {

            double v = 0;

            // System.out.println(Arrays.toString(sample));


            TIntIntHashMap evid = new TIntIntHashMap();

            for (int i = 0; i < sample.length; i++) {
                evid.put(i, sample[i]);
            }

            for (int n : ord) {

                if (isHidden(n)) {
                    continue;
                }

                evid.remove(n);

                // check if variable has hidden parents
                double p;

                if (hasHiddenParents(n)) {
                    BayesianFactor f = vEl.query(n, evid);

                    p = f.potent[sample[n]];
                } else {
                    p = bn.getPotential(n, sample);
                }
                // pf("%d | %.5f \n", n, p);
                v += Math.log10(p);

                if (Double.isNaN(v)) {
                    System.out.println("ciao");
                }

                evid.put(n, sample[n]);
            }

            addLogL(v);
        }

        private boolean isHidden(int n) {
            return (n >= sample.length);
        }

        private boolean hasHiddenParents(int i) {
            for (int p : bn.parents(i)) {
                if (isHidden(p)) {
                    return true;
                }
            }

            return false;
        }
    }

    private void addLogL(double v) {
        synchronized (lock) {
            ll += v;
            done++;
        }
    }
}
