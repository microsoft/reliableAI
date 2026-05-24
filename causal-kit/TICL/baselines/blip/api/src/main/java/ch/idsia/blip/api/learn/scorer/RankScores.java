package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.learn.scorer.RankerScores;
import ch.idsia.blip.core.utils.ParentSet;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


/**
 * Comparison between different scores
 */
public class RankScores extends Api {

    private static final Logger log = Logger.getLogger(
            RankScores.class.getName());

    @Option(name = "-j1", required = true, usage = "First parent set scores output file (jkl format)")
    private static String ph_scores_f = "";

    @Option(name = "-j2", required = true, usage = "Second parent set scores output file (jkl format)")
    private static String ph_scores_s = "";

    private final RankerScores ranker;

    public RankScores() {
        ranker = new RankerScores();
    }

    /**
     * Default command line execution
     */
    public static void main(String[] args) {
        defaultMain(args, new RankScores());
    }

    @Override
    public void exec() throws Exception {

        ParentSet[][] sc1 = null;
        ParentSet[][] sc2 = null;

        sc1 = getScoreReader(ph_scores_f, verbose);
        sc2 = getScoreReader(ph_scores_s, verbose);

        ranker.debug = verbose;

        System.out.println(ranker.execute(sc1, sc2));
    }

}
