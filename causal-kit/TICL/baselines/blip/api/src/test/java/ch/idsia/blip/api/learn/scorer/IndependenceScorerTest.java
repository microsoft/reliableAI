package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.learn.solver.tw.BrutalGreedySolverApi;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.dat.DatFileReader;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.utils.score.Score;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.data.ArrayUtils.hashToParentSet;
import static ch.idsia.blip.core.utils.data.ArrayUtils.parentSetToHash;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class IndependenceScorerTest extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        // oneAndOnlyTrueTest("train-file0", "bic");

        // oneAndOnlyTrueTest("simple", "10000");

        // oneAndOnlyTrueTest("child-50000", "bic");
        oneAndOnlyTrueTest("child-5000", "bic");
        // oneAndOnlyTrueTest("child-50000", "mit");

        // oneAndOnlyTrueTest("accidents.test", "mit");

        // oneAndOnlyTrueTest("child-50000", "bic");

        // oneAndOnlyTrueTest("water-50000", "bic");

        // oneAndOnlyTrueTest("insurance-50000", "bic");

        // oneAndOnlyTrueTest("random2000-50000", "bdeu");

        // oneAndOnlyTrueTest("random10000-50000", "bdeu");

        // oneAndOnlyTrueTest("random10-1", "10240");

        // oneAndOnlyTrueTest("ovarian_61902");


        /*


         oneAndOnlyTrueTest("alarm", "10000");

         oneAndOnlyTrueTest("insurance", "10000");

         oneAndOnlyTrueTest("diabetes", "1000");

         oneAndOnlyTrueTest("diabetes", "10000");

         */

        // oneAndOnlyTrueTest("link", "50000");
    }

    @Test
    public void testFactorial() {
        System.out.println(IndependenceScorer.totNumParentSet(20, 3));
    }

    @Test
    public void testNumberParentSet() {
        System.out.println(IndependenceScorer.totNumParentSet(20, 3));
        System.out.println(IndependenceScorer.totNumParentSet(413, 2));
        System.out.println(IndependenceScorer.totNumParentSet(413, 5));
    }

    private void oneAndOnlyTrueTest(String s, String m) throws IOException, IncorrectCallException {

        String path = basePath + "scorer/" + s;

        int l = 6;

        for (int a = -l; a < l; a++) {

            String file = path + ".dat";
            DatFileReader d = new DatFileReader();

            d.init(file);

            double alpha = Math.pow(2.71, a);

            String scores = String.format("%s-heu-%s-%.3f.scores", path, m,
                    alpha);

            int verbose = 3;
            double max_exec_time = 2;
            int max_pset_size = 4;

            String[] args = {
                "", "-d", file, "-set", scores, "-n",
                String.valueOf(max_pset_size), "-t",
                String.valueOf(max_exec_time), "-pc", m, "-v",
                String.valueOf(verbose), "-pa", String.valueOf(alpha)
            };

            IndependenceScorerApi.main(args);
        }
    }

    @Test
    public void tryScore() throws IOException, IncorrectCallException {

        String[] args = { "", "-set", "/home/loskana/Desktop/random10000.jkl"};

        BrutalGreedySolverApi.main(args);

    }

    @Test
    public void whyyyy() throws IOException, IncorrectCallException {

        int n = 0;

        String path = basePath + "scorer/orig/random2000-0-5000.dat";

        String scores = basePath + "scorer/orig/random2000-0-" + n + ".jkl";

        String m = "bdeu";

        int verbose = 2;
        double max_exec_time = 60;
        int max_pset_size = 6;

        String[] args = {
            "", "-f", path, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose), "-u", String.valueOf(n)};

        IndependenceScorerApi.main(args);

    }

    @Test
    public void testScore2() throws Exception {
        String ph_dat = basePath + "scorer/child-50000.dat";
        DataSet dat = getDataSet(ph_dat);

        // BDeu score = new BDeu(1, dat.n_datapoints, dat.l_n_arity);
        BIC score = new BIC(dat);

        IndependenceScorer is = new IndependenceScorer();

        is.dat = dat;
        is.prepare();
        is.score = score;
        is.max_exec_time = 10;
        is.max_pset_size = 2;
        is.thread_pool_size = 3;
        is.queue_size = 100;
        // independenceScorer.cache_size = 100;

        is.queue_size = Math.min((long) Math.pow(dat.n_var, 3),
                is.max_queue_size);
        // independenceScorer.cache_size = Math.min((long) Math.pow(dat.n_var, 2),
        // independenceScorer.max_cache_size);

        // System.out.println(is.cache_size);
        System.out.println(is.queue_size);

        int n = 0;

        double sk_0 = score.computeScore(n);

        System.out.println(sk_0);

        int[] set_p = { 1, 12, 15};

        double sk_1 = score.computeScore(n, set_p);

        System.out.println(sk_1);

        IndependenceScorer.IndependenceSearcher independenceSearcher = is.getNewSearcher(
                n);

        Thread t = new Thread(independenceSearcher);

        t.start();
        t.join();

    }

    /*

     private void luckTest(String bn, String ds) throws IOException {

     System.out.printf("### On bn: %s%n", bn);

     String path = String.format("%s%s-%s", basePath, bn, ds);

     File f_dat = new File(path + ".dat");
     BufferedReader reader = new BufferedReader(new FileReader(f_dat));
     DataSet dat = getDataSet(reader);

     BDeu bdeu = new BDeu(1, dat.n_datapoints);
     HeuScorer heuScorer = new HeuScorer();

     dat.readMetaData();
     dat.readValuesCache();

     // Compute void scores
     double[] voidScore = new double[dat.n_var];

     for (int n = 0; n < dat.n_var; n++) {
     voidScore[n] = bdeu.computeScore(dat.row_values[n], dat.l_n_arity[n]);
     }

     // Compute one-scores
     double[][] oneScore = new double[dat.n_var][dat.n_var];

     for (int p1 = 0; p1 < dat.n_var; p1++) {
     for (int p2 = 0; p2 < dat.n_var; p2++) {
     if (p1 == p2) {
     continue;
     }

     oneScore[p1][p2] = bdeu.computeScore(dat.sample[p1],
     dat.l_n_arity[p1], dat.row_values[p2], dat.l_n_arity[p2]);
     }
     }

     ArrayList<Boolean> bet = new ArrayList<Boolean>();
     ArrayList<Double> diff = new ArrayList<Double>();
     ParentSetCache cache = new ParentSetCache(dat.n_var * 10);

     for (int n = 0; n < dat.n_var; n++) {
     System.out.println("Doing variable... " + n);
     luckVariable(n, heuScorer, bdeu, dat, cache, bet, diff, oneScore);
     }

     int n = 0;

     for (Boolean pb : bet) {
     if (pb) {
     n++;
     }
     }
     System.out.println(String.format("%.3f", (n * 1.0) / bet.size()));
     System.out.printf("Bet: %d - Discarded: %d%n", bet.size(), diff.size());

     }

     private void luckVariable(int n, HeuScorer.HeuSearcher heuScorer, BDeu bdeu, DataFileReader dat, ParentSetCache cache, List<Boolean> bet, ArrayList<Double> diff, double[][] oneScore) {

     TIntArrayList bestPSet = null;
     double bestScore = Math.pow(2, 30);

     // Best call for two-parent set
     for (int p1 = 0; p1 < dat.n_var; p1++) {
     if (p1 == n) {
     continue;
     }

     TIntArrayList p = new TIntArrayList();

     p.add(p1);
     long hash_p = heuScorer.parentSetToHash(p.toArray());

     evaluateHeuristic(n, heuScorer, bdeu, dat, cache, bet, diff, oneScore[n],
     p1, p, hash_p);
     }

     // Best call for three-parent set
     for (int p1 = 0; p1 < dat.n_var; p1++) {
     if (p1 == n) {
     continue;
     }

     for (int p2 = p1 + 1; p2 < dat.n_var; p2++) {
     if (p2 == n) {
     continue;
     }

     TIntArrayList p = new TIntArrayList();

     p.add(p1);
     p.add(p2);
     long hash_p = heuScorer.parentSetToHash(p.toArray());

     evaluateHeuristic(n, heuScorer, bdeu, dat, cache, bet, diff,
     oneScore[n], p2, p, hash_p);
     }
     }

     }
     */

    @Test
    public void testHashing() throws IOException {

        int n_var = 20;

        IndependenceScorer independenceScorer = new IndependenceScorer();

        independenceScorer.dat = null;
        independenceScorer.dat.n_var = n_var;

        List<Long> listHash = new ArrayList<Long>();

        checkHash(independenceScorer, "[1, 4, 6]", listHash, n_var);

        checkHash(independenceScorer, "[5]", listHash, n_var);

        checkHash(independenceScorer, "[1, 4, 7]", listHash, n_var);

        checkHash(independenceScorer, "[2, 3, 5]", listHash, n_var);

        checkHash(independenceScorer, "[0, 9, 11]", listHash, n_var);

        checkHash(independenceScorer, "[0, 11]", listHash, n_var);

        checkHash(independenceScorer, "[2, 4]", listHash, n_var);

        checkHash(independenceScorer, "[0, 2, 4]", listHash, n_var);

        System.out.println(Arrays.toString(hashToParentSet(0, n_var)));

        System.out.println(Arrays.toString(hashToParentSet(1, n_var)));

        System.out.println(Arrays.toString(hashToParentSet(20, n_var)));
        System.out.println(Arrays.toString(hashToParentSet(21, n_var)));
        System.out.println(Arrays.toString(hashToParentSet(22, n_var)));

        System.out.println(getHash("2 4", n_var));

        System.out.println(getHash("0 2 4", n_var));

    }

    private long getHash(String s, int n_var) {
        TIntArrayList t = new TIntArrayList();

        s = s.replace("[", "").replace("]", "").replace(",", "");
        for (String p : s.split(" ")) {
            t.add(Integer.parseInt(p));
        }
        t.sort();
        return parentSetToHash(t.toArray(), n_var);
    }

    private void checkHash(IndependenceScorer independenceScorer, String s, List<Long> listHash, int n_var) {

        long hash = getHash(s, n_var);
        int[] new_test = hashToParentSet(hash, n_var);

        assertEquals(s, Arrays.toString(new_test));

        assertTrue(!listHash.contains(hash));
        listHash.add(hash);
    }

    @Test
    public void testParentSet() throws IOException {

        int n_var = 20;

        IndependenceScorer independenceScorer = new IndependenceScorer();

        independenceScorer.dat = null;
        independenceScorer.dat.n_var = n_var;

        int[] pset = { 2, 4, 8};

        assertEquals("[4, 8]", Arrays.toString(ArrayUtils.reduceArray(pset, 2)));
        assertEquals("[2, 8]", Arrays.toString(ArrayUtils.reduceArray(pset, 4)));
        assertEquals("[2, 4]", Arrays.toString(ArrayUtils.reduceArray(pset, 8)));

        assertEquals("[1, 2, 4, 8]",
                Arrays.toString(ArrayUtils.expandArray(pset, 1)));
        assertEquals("[2, 3, 4, 8]",
                Arrays.toString(ArrayUtils.expandArray(pset, 3)));
        assertEquals("[2, 4, 7, 8]",
                Arrays.toString(ArrayUtils.expandArray(pset, 7)));
        assertEquals("[2, 4, 8, 10]",
                Arrays.toString(ArrayUtils.expandArray(pset, 10)));
    }

    @Test
    public void testBdeu() throws IncorrectCallException, IOException {

        String ph_dat = basePath + "scorer/child-50000.dat";
        DataSet dat = getDataSet(ph_dat);

        BIC score = new BIC(dat);
        int n = 0;
        HelpScorer h = new HelpScorer(score, dat, n);

        System.out.println("sk: " + score.computeScore(n));

        System.out.println("sk_11: " + h.s(11));

        h.cmp(15, 18);
        h.cmp(5, 9);

    }

    private class HelpScorer {
        private final int n;
        private final DataSet dat;
        private final Score score;

        public HelpScorer(Score score, DataSet dat, int n) {
            this.score = score;
            this.dat = dat;
            this.n = n;
        }

        public double s(int i) {
            return s(new int[] { i});
        }

        private double s(int[] set_p) {
            int[][] p_values = score.computeParentSetValues(set_p);

            return score.computeScore(n, set_p);
        }

        public double s(int i, int j) {
            int[] a = new int[] { i, j};

            Arrays.sort(a);
            return s(a);
        }

        public void cmp(int i, int j) {

            System.out.printf("sk_%d,%d: %.2f / %.2f \n", i, j, s(i, j),
                    s(i) + s(j));
        }
    }
}
