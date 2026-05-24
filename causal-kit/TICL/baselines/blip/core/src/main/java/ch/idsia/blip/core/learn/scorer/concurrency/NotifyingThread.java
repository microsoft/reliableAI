package ch.idsia.blip.core.learn.scorer.concurrency;


import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;


public class NotifyingThread extends Thread {

    private final Set<ThreadCompleteListener> listeners = new CopyOnWriteArraySet<ThreadCompleteListener>();

    public NotifyingThread(Runnable r) {
        super(r);
    }

    public final void addListener(final ThreadCompleteListener listener) {
        listeners.add(listener);
    }

    public final void removeListener(final ThreadCompleteListener listener) {
        listeners.remove(listener);
    }

    private void notifyListeners() {
        for (ThreadCompleteListener listener : listeners) {
            listener.notifyOfThreadComplete();
        }
    }

    @Override
    public final void run() {
        try {
            long start = System.currentTimeMillis();

            super.run();

            /*
             System.out.printf("Done in: %.1f \n",
             (System.currentTimeMillis() - initCl) / 1000.0);
             */

        } finally {
            notifyListeners();
        }
    }

}
