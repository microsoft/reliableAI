package ch.idsia.blip.api;


import ch.idsia.blip.api.learn.scorer.RankScores;
import ch.idsia.blip.core.learn.scorer.RankerScores;
import ch.idsia.blip.core.utils.ParentSet;
import org.junit.Test;

import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


public class RankScoreTest extends TheTest {

    @Test
    public void testComplete() throws Exception {

        String[] args = new String[] {
            "ranker", "-set", "../experiments/scorer/win95pts/", "-v", "2"};

        RankScores.main(args);
    }

    @Test
    public void testDetailed() throws Exception {

        String s = "../experiments/scorer/win95pts/60/";

        ParentSet[][] sc1 = getScoreReader(s + "heu.scores", 2);
        ParentSet[][] sc2 = getScoreReader(s + "greedy.scores", 2);

        RankerScores ranker = new RankerScores();

        ranker.debug = 2;
        ranker.execute(sc1, sc2);
    }

    @Test
    public void testDetailed2() throws Exception {

        String s = "../experiments/scorer/win95pts/60/";

        ParentSet[][] sc1 = getScoreReader(s + "heu.scores", 2);
        ParentSet[][] sc2 = getScoreReader(s + "seq.scores", 2);

        RankerScores ranker = new RankerScores();

        ranker.debug = 2;
        ranker.execute(sc1, sc2);
    }

}
