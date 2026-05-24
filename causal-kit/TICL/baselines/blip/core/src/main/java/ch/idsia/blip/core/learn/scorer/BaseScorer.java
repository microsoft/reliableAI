package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.*;
import ch.idsia.blip.core.learn.scorer.concurrency.Executor;
import ch.idsia.blip.core.learn.scorer.utils.OpenParentSet;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntDoubleHashMap;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.data.map.ArrayHashingStrategy;
import ch.idsia.blip.core.utils.data.map.TCustomHashMap;

import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public abstract class BaseScorer extends App {

    private static final Logger log = Logger.getLogger(
            BaseScorer.class.getName());

    // Input datapoint file
    public DataSet dat;

    // Scores file
    public String ph_scores;

    // BDeu scorer
    public Score score;

    // Parents size limit (for memory!)
    protected final long max_parents = (long) Math.pow(2, 8);

    // Equivalent sample size
    public double alpha;

    // Maximum in-degree for variable
    public int max_pset_size;

    // Name of the score to be used
    public String scoreNm = "bdeu";

    // Flag to graph name of variables, instead of index
    protected boolean write_names;

    // Flag for searching only some variables
    public String choice_variables;

    public int n_var;

    protected ScoreWriter scoreWriter;

    public double max_searcher_time;

    protected abstract String getName();

    @Override
    public void prepare() {
        super.prepare();

        n_var = dat.n_var;
    }

    public void go(String dat_path) throws Exception {
        start = System.currentTimeMillis();
        if (verbose > 0) {
            logf("Reading from datafile '%s'... \n", dat_path);
        }
        go(getDataSet(dat_path));
    }

    public void go(DataSet in_dat) throws Exception {

        dat = in_dat;

        score.dat = dat;

        prepare();

        if (choice_variables != null && !"".equals(choice_variables)) {
            searchChoice();
        } else {
            searchAll();
        }

        if (verbose > 0) {
            pf("... done - eval: %d (el: %.2f) \n", score.numEvaluated,
                    (System.currentTimeMillis() - start) / 1000.0);
        }
    }

    private void searchChoice() throws Exception {

        int s = 0, e = 0;

        Pattern p = Pattern.compile("\\d+");
        Matcher m = p.matcher(choice_variables);

        if (m.matches()) {
            s = Integer.valueOf(choice_variables);
            e = s + 1;
        }

        p = Pattern.compile("(\\d+)-(\\d+)");
        m = p.matcher(choice_variables);
        if (m.matches()) {
            s = Integer.valueOf(m.group(1));
            e = Math.min(n_var, Integer.valueOf(m.group(2)) + 1);
        }

        set_max_exec_time(e-s);

        Thread t1 = new Thread(new Executor(thread_pool_size, s, e, this));

        scoreWriter = new ScoreWriter(this, ph_scores, s, e, verbose);
        Thread t2 = new Thread(scoreWriter);

        t1.start();
        t2.start();

        t1.join();
        t2.join();

    }

    /*
     private void searchOne(int n) throws InterruptedException, UnsupportedEncodingException, FileNotFoundException {

     // available time
     if (max_exec_time == 0)
     max_searcher_time = 60;

     Runnable r = getNewSearcher(n);
     Thread t = new Thread(r);

     scoreWriter = new ScoreWriter(this, ph_scores, n, n+1, verbose);
     Thread t2 = new Thread(scoreWriter);

     t.start();
     t2.start();

     t.join();
     t2.join();
     }
     */

    public void searchAll() throws InterruptedException, IOException {

        set_max_exec_time(dat.n_var);

        if (verbose > 0) {
            pf("Executing with: \n");
            pf("%-12s: %s \n", "code", this.getClass().getName());
            pf("%-12s: %d \n", "threads", thread_pool_size);
            pf("%-12s: %.2f \n", "search_time", max_searcher_time);
            pf("%-12s: %d \n", "max_degree", max_pset_size);
        }

        if (verbose > 1) {
            pf("... searching (el: %.2f) \n",
                    (System.currentTimeMillis() - start) / 1000.0);
        }

        Thread t1 = new Thread(new Executor(thread_pool_size, 0, n_var, this));

        scoreWriter = new ScoreWriter(this, ph_scores, 0, n_var, verbose);
        Thread t2 = new Thread(scoreWriter);

        if (verbose > 0) {
            pf("... writing down:");
        }

        t1.start();
        t2.start();

        t1.join();
        t2.join();

        conclude();

    }

    private void set_max_exec_time(int n) {

        // available time
        if (max_exec_time > 0) {
            max_exec_time -= (System.currentTimeMillis() - start) / 1000.0;
            // for each searcher, (total time) * (num of threads) / (num of variables)
            max_searcher_time = (max_exec_time * Math.min(n, thread_pool_size)) / n;
        } else {
            max_searcher_time = 60;
        }
    }

    protected void conclude() throws IOException {
        if (verbose > 0) {
            pf("done! \n");
        }

        scoreWriter.wr.close();
    }

    public abstract BaseSearcher getNewSearcher(int n);

    /**
     * Get a string that represent the set score
     *
     * @param sk    score
     * @param set_p parent set
     * @return representation
     */
    String strPSetScore(Double sk, int[] set_p) {

        String mex = String.format("%.4f %d", sk, set_p.length);

        for (int p : set_p) {
            mex += String.format(" %s", getVarName(p));
        }
        return mex;
    }

    /**
     * @param n variable of interest
     * @return name of variable to graph (index or actual name, depending on flag)
     */
    private String getVarName(int n) {
        if (write_names) {
            return dat.l_nm_var[n];
        } else {
            return String.valueOf(n);
        }
    }

    /**
     * Check if the skore has to be pruned: exists a subset with an higher skore
     *
     * @param sk     skore to check
     * @param set_p  parents in the set to check
     * @param scores scores list
     * @return if there is a better skore in the subsets
     */
    protected boolean checkToPrune(double sk, int[] set_p, Map<int[], Double> scores) {
        for (int p : set_p) {
            int[] set_new = reduceArray(set_p, p);

            if (scores.containsKey(set_new) && (sk <= scores.get(set_new))) {
                return true;
            }

            if (set_new.length > 1) {
                if (checkToPrune(sk, set_new, scores)) {
                    return true;
                }
            }
        }

        return false;
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);

        this.ph_scores = gStr("ph_scores");
        this.max_pset_size = gInt("max_pset_size", 6);
        this.scoreNm = gStr("scoreNm", "bdeu");
        this.alpha = gDouble("alpha", 1.0);
        this.choice_variables = gStr("choice_variables", "");

        scoreNm = scoreNm.toLowerCase().trim();

        if ("bdeu".equals(scoreNm)) {
            score = new BDeu(alpha, dat);
        } else if ("bdeu2".equals(scoreNm)) {
            score = new BDeu2(alpha, dat);
        } else if ("k2".equals(scoreNm)) {
            score = new K2(dat);
        } else if ("bic".equals(scoreNm)) {
            score = new BIC(dat);
        } else if ("mit".equals(scoreNm)) {
            score = new MIT(alpha, dat);
        } else {
            score = new BDeu(alpha, dat);
            pf("Unknown score: %s \n", scoreNm);
        }
    }

    class ScoreWriter implements Runnable {

        private final int end;
        private final int st;
        private final int verbose;

        private final Writer wr;

        private Writer cacheWr;

        private HashMap<Integer, Map<int[], Double>> cache;

        ScoreWriter(BaseScorer sc, String ph_scores, int start, int end, int verbose) {

            wr = getWriter(ph_scores);
            this.st = start;
            this.end = Math.min(end, n_var + 1);

            try {
                preamble(sc, wr);
                wr.flush();

            } catch (IOException e) {
                logExp(log, e);
            }

            cache = new HashMap<Integer, Map<int[], Double>>();
            this.verbose = verbose;
        }

        public void run() {

            int i = st;
            boolean cnt;

            while (i < end) {

                cnt = cache.containsKey(i);

                if (!cnt) {
                    waitAsec(1000);
                    continue;
                }

                if (verbose > 1) {
                    pf("%d (el: %.2f), ", i,
                            (System.currentTimeMillis() - start) / 1000.0);
                } else if (verbose > 0) {
                    pf("%d, ", i);
                }

                writeScores(wr, i, cache.get(i));

                cache.remove(i);

                i++;

            }
        }

        private void waitAsec(int s) {
            try {
                Thread.sleep(s);
            } catch (InterruptedException e) {
                logExp(log, e);
            }
        }

        // add a new variables scores to graph down
        public void add(int n, Map<int[], Double> scores) {
            synchronized (lock) {
                cache.put(n, scores);
            }
        }

    }

    protected Map<SIntSet, Double> pruneScores(Map<int[], Double> scores) {

        Double voidSk = scores.get(new int[0]);

        Map<SIntSet, Double> new_scores = new TreeMap<SIntSet, Double>();

        // Pick only the best scores
        for (int[] pset : RandomStuff.sortInvByValues(scores).keySet()) {

            // Check when to limit scores
            double sk = scores.get(pset);

            // log.conclude(set + " " + new_sk);

            // Decide to put it in the final score
            boolean toPrune = sk < voidSk + 0.0001;

            toPrune = toPrune || checkToPrune(sk, pset, scores);

            if (!toPrune) {
                new_scores.put(new SIntSet(pset), sk);
            }
        }

        new_scores.put(new SIntSet(), voidSk);

        return new_scores;
    }

    protected void writeScores(Writer wr, int n, Map<int[], Double> scores) {

        // prune scores
        Map<SIntSet, Double> prScores = pruneScores(scores);

        try {

            wf(wr, "%s %d\n", getVarName(n), prScores.size());

            Set<SIntSet> aux = RandomStuff.sortInvByValues(prScores).keySet();

            for (SIntSet pset : aux) {

                wr.write(strPSetScore(scores.get(pset.set), pset.set));
                wr.write("\n");
            }

            wr.flush();

        } catch (IOException e) {
            log.severe(
                    String.format("Error writing score to file: %s",
                    e.getMessage()));
        }

    }

    public abstract class BaseSearcher implements Runnable {

        /**
         * Variable to work with
         */
        protected final int n;

        /**
         * List of good parents for the variable
         */
        int[] parents;

        double voidSk;

        protected TIntDoubleHashMap oneScores;

        protected Map<int[], Double> scores;

        double m_elapsed;

        double m_start;

        BaseSearcher(int in_n) {
            n = in_n;
        }

        void addScore(double sk) {
            addScore(new int[0], sk);
        }

        void addScore(int p, double sk) {
            addScore(new int[] { p}, sk);
        }

        protected void addScore(int[] p, double sk) {
            scores.put(p, sk);
        }

        void prepare() {

            m_start = System.currentTimeMillis();

            // Prepare scores
            scores = new TCustomHashMap<int[], Double>(
                    new ArrayHashingStrategy());

            // Void score
            voidSk = score.computeScore(n);
            addScore(voidSk);

            // One score
            searchSingleParents(n);
        }

        /*
         private void evaluateParents(int n) {

         if (monovalue(n)) {
         parents = new int[0];
         return;
         }

         // MutualInformation mi = new MutualInformation(dat, 0.999, 10);
         TIntArrayList l = new TIntArrayList();

         for (int n2 = 0; n2 < n_var; n2++) {

         if (n == n2) {
         continue;
         }

         if (monovalue(n2))
         continue;


         if (mi.condInd(n, n2))
         continue;


         l.add(n2);
         }

         parents = l.toArray();

         }
         */

        private boolean monovalue(int n2) {
            for (int v = 0; v < dat.l_n_arity[n2]; v++) {
                if (dat.row_values[n2][v].length * 1.0 / dat.n_datapoints
                        > 0.9999) {
                    return true;
                }
            }
            return false;
        }

        private void searchSingleParents(int n) {

            oneScores = new TIntDoubleHashMap();

            // if (monovalue(n)) {
            // parents = new int[0];
            // return;
            // }

            double worstQueueScore = 0;

            TreeSet<OpenParentSet> open = new TreeSet<OpenParentSet>();

            for (int n2 = 0; n2 < n_var; n2++) {

                if (n == n2) {
                    continue;
                }

                // if (monovalue(n2))
                // continue;

                double oneSk = score.computeScore(n, n2);
                if (Math.abs(oneSk) < 0.00000001)
                    continue;

                oneScores.put(n2, oneSk);
                addScore(n2, oneSk);

                boolean toDropWorst = false;

                if (open.size() > max_parents) {

                    if (oneSk < worstQueueScore) {
                        // log.conclude("pruned");
                        continue;
                    }

                    toDropWorst = true;
                }

                // Drop worst element in queue, to make room!
                if (toDropWorst) {
                    open.pollLast();
                    worstQueueScore = open.last().sk;
                } else // If we didn't drop any element, check if we have to update the current
                // worst score!
                if (oneSk < worstQueueScore) {
                    worstQueueScore = oneSk;
                }

                open.add(new OpenParentSet(n2, -1, oneSk, null));
            }
            TIntArrayList l = new TIntArrayList(open.size());

            for (OpenParentSet p : open) {
                int n2 = p.s[0];

                l.add(n2);
            }

            parents = l.toArray();
            Arrays.sort(parents);
        }

        protected void conclude() {

            // If parent writer is ready, write there
            if (scoreWriter != null) {
                scoreWriter.add(n, scores);
            }
        }

        boolean thereIsTime() {
            if (max_searcher_time == 0) {
                return true;
            }
            m_elapsed = ((System.currentTimeMillis() - m_start) / 1000.0);
            return m_elapsed < max_searcher_time;
        }

    }

    protected void preamble(BaseScorer sc, Writer wr) throws IOException {
        wf(wr, "# Method: %s \n", sc.getName(), max_searcher_time);
        wf(wr, "# Score function: %s \n", score.descr());
        wf(wr, "# Max in-degree: %d \n", max_pset_size);
        wf(wr, "%d\n", sc.n_var);
    }
}

