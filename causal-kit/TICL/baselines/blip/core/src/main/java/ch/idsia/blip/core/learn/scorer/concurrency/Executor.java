package ch.idsia.blip.core.learn.scorer.concurrency;


import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.utils.RandomStuff;

import java.util.logging.Logger;


public class Executor implements Runnable, ThreadCompleteListener {

    private static final Object lock = new Object();

    /**
     * Logger
     */
    private static final Logger log = Logger.getLogger(Executor.class.getName());
    private final int max_thread;
    private final BaseScorer scorer;
    private final int start;
    private final int end;
    private Integer crt_thread;

    public Executor(int max_thread, int start, int end, BaseScorer scorer) {
        this.max_thread = max_thread;
        this.scorer = scorer;
        this.start = start;
        this.end = end;
    }

    @Override
    public void run() {

        crt_thread = 0;

        int v = start;

        while (v < end) {

            try {
                synchronized (lock) {
                    if (crt_thread < max_thread) {
                        crt_thread++;
                        Runnable r = scorer.getNewSearcher(v);
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
    }

    @Override
    public void notifyOfThreadComplete() {
        synchronized (lock) {
            crt_thread--;
            lock.notify();
        }
    }
}
