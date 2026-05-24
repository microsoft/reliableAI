package ch.idsia.blip.core.learn.scorer;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.other.ParentSetCache;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.parentSetToHash;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static org.junit.Assert.assertTrue;


public class RankerScoresTest extends TheTest {

    private static final Logger log = Logger.getLogger(
            RankerScoresTest.class.getName());

    @Test
    public void testRankerScore() throws Exception {

        String sc = basePath + "scorer/diabetes/5/";

        String ph1 = sc + "heu.scores";
        String ph2 = sc + "seq.scores";

        ParentSet[][] sc1 = RandomStuff.getScoreReader(ph1, 1);
        ParentSet[][] sc2 = RandomStuff.getScoreReader(ph2, 1);

        RankerScores ranker = new RankerScores();

        ranker.debug = 2;
        System.out.println("\n" + ranker.execute(sc1, sc2));
    }

    @Test
    public void testScore() throws Exception {
        String ph_dat = basePath + "scorer/hepar2/d-100.dat";
        DataSet dat = getDataSet(ph_dat);

        BDeu score = new BDeu(1, dat);

        IndependenceScorer us = new IndependenceScorer();

        us.dat = dat;
        us.prepare();
        us.score = score;
        us.max_exec_time = 3;
        us.max_pset_size = 3;
        us.thread_pool_size = 3;
        us.queue_size = 100;
        // us.cache_size = 100;

        int n = 1;
        int[] set_p = new int[] { 20, 26, 31};
        int p2 = 31;

        IndependenceScorer.IndependenceSearcher independenceSearcher = us.getNewSearcher(
                1);

        ParentSetCache cache = new ParentSetCache(100);

        // FIRST WAY, NO CACHE
        int[][] p_values = score.computeParentSetValues(set_p);

        double sk_1 = score.computeScore(n, p_values.length);

        System.out.println(sk_1);

        // SECOND WAY, CACHE
        int[] subset = new int[] { 20, 31};

        assertTrue(parentSetToHash(subset, us.n_var) == 2190);

        p_values = score.computeParentSetValues(subset);
        System.out.println("prev: " + p_values.length);
        cache.save(new SIntSet(subset), p_values);

        /*
         p_values = independenceSearcher.getParentSetValues(
         new OpenParentSet(new SIntSet(subset), p2, 0));
         assertTrue(us.hit_ok == 1);
         assertTrue(us.hit_ko == 0);
         */

        double sk_2 = score.computeScore(n, subset);

        System.out.println(sk_2);

        // THIRD WAY: RUN!

        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(System.out));
        Thread t = new Thread(independenceSearcher);

        t.start();
        t.join();

        double sk_3 = independenceSearcher.scores.get(new SIntSet(set_p));

        System.out.println(sk_3);

        assertTrue(sk_1 == sk_2);
        assertTrue(sk_2 == sk_3);
    }

    @Test
    public void testScore2() throws Exception {
        String ph_dat = basePath + "scorer/win95pts/d-100.dat";
        DataSet dat = getDataSet(ph_dat);

        BDeu score = new BDeu(1, dat);

        IndependenceScorer is = new IndependenceScorer();

        is.dat = dat;
        is.prepare();
        is.score = score;
        is.max_exec_time = 3;
        is.max_pset_size = 3;
        is.thread_pool_size = 3;
        is.queue_size = 100;
        // is.cache_size = 100;

        int n = 2;
        int[] set_p = new int[] { 12, 40, 61};

        IndependenceScorer.IndependenceSearcher independenceSearcher = is.getNewSearcher(
                2);

        ParentSetCache cache = new ParentSetCache(100);

        // FIRST WAY, NO CACHE
        int[][] p_values = score.computeParentSetValues(set_p);

        double sk_1 = score.computeScore(n, set_p);

        System.out.println(sk_1);

        // SECOND WAY, CACHE
        int[] subset = new int[] { 12, 40};
        int p2 = 61;

        p_values = score.computeParentSetValues(subset);
        System.out.println("prev: " + p_values.length);
        cache.save(new SIntSet(subset), p_values);

        /*
         p_values = independenceSearcher.getParentSetValues(
         new OpenParentSet(new SIntSet(subset), p2, 0));
         assertTrue(is.hit_ok == 1);
         assertTrue(is.hit_ko == 0);
         */
        double sk_2 = score.computeScore(n, subset);

        System.out.println(sk_2);

        // THIRD WAY: RUN!

        Thread t = new Thread(independenceSearcher);

        t.start();
        t.join();

        double sk_3 = independenceSearcher.scores.get(new SIntSet(set_p));

        System.out.println(sk_3);

        // Fourth way: SEQ

        SeqScorer seqScorer = new SeqScorer();
        SeqScorer.SeqSearcher seqSearcher = seqScorer.getNewSearcher(2);

        t = new Thread(independenceSearcher);
        t.start();
        t.join();

        double sk_4 = independenceSearcher.scores.get(new SIntSet(set_p));

        System.out.println(sk_4);

        assertTrue(sk_1 == sk_2);
        assertTrue(sk_2 == sk_3);
        assertTrue(sk_3 == sk_4);
    }

    @Test
    public void testHaresTortoise() throws Exception {

        ParentSet[] hares = new ParentSet[6];
        ParentSet[] tortoise = new ParentSet[6];

        int sk = 1;
        RankerScores ranker = new RankerScores();

        ranker.debug = 2;
        int n = 10;

        tortoise[0] = new ParentSet(-sk, ArrayUtils.hashToParentSet(sk++, n));

        for (int i = 0; i < 5; i++) {
            hares[i] = new ParentSet(-sk, ArrayUtils.hashToParentSet(sk++, n));
        }

        for (int i = 1; i < 6; i++) {
            tortoise[i] = new ParentSet(-sk, ArrayUtils.hashToParentSet(sk++, n));
        }

        hares[5] = new ParentSet(-sk, ArrayUtils.hashToParentSet(sk++, n));

        System.out.println(Arrays.toString(tortoise));
        System.out.println(Arrays.toString(hares));

        Pair<Double, Double> res = ranker.compare(tortoise, hares);

        System.out.println(res);
        assertTrue(res.getFirst() == 11);
        assertTrue(res.getSecond() == 25);
    }
}
