package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.learn.scorer.GreedyScorer;

import java.util.logging.Logger;


public class GreedyScorerApi extends ScorerApi {

    public static void main(String[] args) {
        defaultMain(args, new GreedyScorerApi());
    }

    @Override
    protected BaseScorer getScorer() {
        return new GreedyScorer();
    }
}
