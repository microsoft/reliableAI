package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.learn.scorer.SeqScorer;

import java.util.logging.Logger;


public class SeqScorerApi extends ScorerApi {

    private static final Logger log = Logger.getLogger(
            SeqScorerApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new SeqScorerApi());
    }

    @Override
    protected BaseScorer getScorer() {
        return new SeqScorer();
    }
}
