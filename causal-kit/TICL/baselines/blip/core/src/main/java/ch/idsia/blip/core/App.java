package ch.idsia.blip.core;


import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Random;


public class App extends Base {

    public long start;

    // Maximum execution time
    public double max_exec_time;

    // Maximum number of threads to use
    public int thread_pool_size;

    // Lock for concurrent searchers
    public final Object lock = new Object();

    public int verbose;

    public Writer logWr;

    public long seed;

    public Random rand;

    private HashMap<String, String> options;

    public App() {
        start = System.currentTimeMillis();
    }

    public void logf(String format, Object... args) {
        log(String.format(format, args));
    }

    public void safeLogf(String format, Object... args) {
        synchronized (lock) {
            logf(format, args);
        }
    }

    public void log(String s) {
        try {
            if (logWr != null) {
                logWr.write(s);
                logWr.flush();
            } else {
                System.out.print(s);
            }
            System.out.flush();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    protected void prepare() {

        if (seed == 0) {
            seed = System.currentTimeMillis();
        }

        rand = new Random(seed);

        if (thread_pool_size == 0) {
            thread_pool_size = Runtime.getRuntime().availableProcessors();
        }

    }

    public int randInt(int min, int max) {
        return rand.nextInt((max - min) + 1) + min;
    }

    public double randProb() {
        return rand.nextDouble();
    }

    public double randDouble() {
        return rand.nextDouble();
    }

    protected String gStr(String k) {
        return options.get(k);
    }

    protected int gInt(String k) {
        return Integer.valueOf(options.get(k));
    }

    protected int gInt(String k, int v) {
        if (options.containsKey(k)) {
            return Integer.valueOf(options.get(k));
        } else {
            return v;
        }
    }

    protected double gDouble(String k, double v) {
        if (options.containsKey(k)) {
            return Double.valueOf(options.get(k));
        } else {
            return v;
        }
    }

    protected String gStr(String k, String v) {
        if (options.containsKey(k)) {
            return options.get(k);
        } else {
            return v;
        }
    }

    protected boolean gBool(String k) {
        if (options.containsKey(k)) {
            return Boolean.valueOf(options.get(k));
        }
        return false;
    }

    public void init(HashMap<String, String> options) {
        this.options = options;

        seed = gInt("seed", 0);
        thread_pool_size = gInt("thread_pool_size", 0);
        max_exec_time = gDouble("max_exec_time", 10);
        verbose = gInt("verbose", 0);
    }

    public void init() {
        this.init(new HashMap<String, String>());
    }

    long getAvailableMemory() {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory(); // current heap allocated to the VM process
        long freeMemory = runtime.freeMemory(); // out of the current heap, how much is free
        long maxMemory = runtime.maxMemory(); // Max heap VM can use e.g. Xmx setting
        long usedMemory = totalMemory - freeMemory; // how much of the current heap the VM is using
        long availableMemory = maxMemory - usedMemory; // available memory i.e. Maximum heap size minus the current amount used

        return availableMemory;
    }

}
