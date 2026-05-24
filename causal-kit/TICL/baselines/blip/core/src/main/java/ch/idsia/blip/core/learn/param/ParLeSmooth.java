package ch.idsia.blip.core.learn.param;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.learn.scorer.concurrency.NotifyingThread;
import ch.idsia.blip.core.learn.scorer.concurrency.ThreadCompleteListener;
import ch.idsia.blip.core.common.LLEval;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSetReader;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class ParLeSmooth extends App {

    private static final Logger log = Logger.getLogger(
            ParLeSmooth.class.getName());

    private double[] scale = new double[] {
        0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100
    };

    public double highestLL;

    protected BayesianNetwork best;

    protected String valid;

    protected DataSet train;

    protected BayesianNetwork res;

    public double highestA;

    public BayesianNetwork go(BayesianNetwork res, DataSet train, String valid) throws FileNotFoundException {

        prepare(res, train, valid);

        Thread t1 = new Thread(new SmoothExecutor(thread_pool_size, this));

        t1.start();
        try {
            t1.join();
        } catch (InterruptedException e) {
            logExp(log, e);
        }

        return best;
    }

    protected void prepare(BayesianNetwork res, DataSet train, String valid) {
        highestLL = -Double.MAX_VALUE;
        best = null;

        this.res = res;
        this.train = train;
        this.valid = valid;
    }

    private class SmoothExecutor implements Runnable, ThreadCompleteListener {

        private final Object lock = new Object();
        private final int max_thread;
        private final ParLeSmooth smooth;
        private Integer crt_thread;
        private int completed;

        public SmoothExecutor(Integer max_thread, ParLeSmooth smooth) {
            this.max_thread = max_thread;
            this.smooth = smooth;
        }

        @Override
        public void run() {

            completed = 0;

            crt_thread = 0;
            int v = 0;

            while (v < smooth.scale.length) {

                try {
                    synchronized (lock) {
                        if (crt_thread < max_thread) {
                            crt_thread++;
                            Runnable r = smooth.getNewSearcher(smooth.scale[v]);
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

            while (completed != smooth.scale.length) {

                // p(completed);
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

    private Runnable getNewSearcher(double a) {
        return new SmoothSearcher(a);
    }

    private class SmoothSearcher implements Runnable {

        private final double alpha;

        public SmoothSearcher(double a) {
            this.alpha = a;
        }

        @Override
        public void run() {
            ParLeBayes p = new ParLeBayes(alpha);
            BayesianNetwork newBn = p.go(res, train);
            LLEval l = new LLEval();

            l.go(newBn, getDataSetReader(valid));
            if (verbose > 0) {
                logf("Propose new ll: %.4f for alpha: %.4f \n", l.ll, alpha);
            }
            propose(l.ll, newBn, alpha);
        }
    }

    protected void propose(double ll, BayesianNetwork newBn, double alpha) {
        synchronized (lock) {
            if (ll > highestLL) {
                highestLL = ll;
                best = newBn;
                highestA = alpha;
            }
        }
    }
}
