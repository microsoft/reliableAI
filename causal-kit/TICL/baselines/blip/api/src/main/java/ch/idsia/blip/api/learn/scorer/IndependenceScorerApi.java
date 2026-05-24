package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;

import java.util.logging.Logger;


public class IndependenceScorerApi extends ScorerApi {

    public static void main(String[] args) {
        defaultMain(args, new IndependenceScorerApi());
    }

    @Override
    protected BaseScorer getScorer() {
        return new IndependenceScorer();
    }
}
