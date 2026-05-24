package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.learn.scorer.SeqAdvScorer;

import java.util.logging.Logger;


public class SeqAdvScorerApi extends ScorerApi {

    private static final Logger log = Logger.getLogger(
            SeqAdvScorerApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new SeqAdvScorerApi());
    }

    @Override
    protected BaseScorer getScorer() {
        return new SeqAdvScorer();
    }
}
