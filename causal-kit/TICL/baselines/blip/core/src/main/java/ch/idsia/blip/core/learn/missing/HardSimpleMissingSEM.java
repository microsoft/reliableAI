package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.utils.ParentSet;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class HardSimpleMissingSEM extends HardMissingSEM {

    @Override
    protected BayesianNetwork learn()
            throws Exception {
        String base = f("%s/%s-%d", path, "em", cnt);

        String jkl = base + ".jkl";

        IndependenceScorer is = new IndependenceScorer();

        int pset_time = Math.min(time_for_variable * dat.n_var, 1200);
        p(pset_time);
        prp(is, jkl, 1, pset_time, "bdeu", 1.0D,
                thread_pool_size);
        is.verbose = 0;
        DataSet d = setScore(is);

        is.go(d);

        String res = base + ".res";
        ParentSet[][] sc = new ScoreReader(jkl).readScores();

        WinAsobsSolver gs = new WinAsobsSolver();

        int brt_time = Math.min(time_for_variable * dat.n_var / 10, 600);
        prp(gs, res, sc, brt_time, thread_pool_size);
        gs.tryFirst = bests;
        gs.out_solutions = 10;
        gs.write_solutions = false;
        gs.verbose = 1;
        gs.go();

        bests.addAll(gs.open);

        BayesianNetwork newbn = BnResReader.ex(res);

        newbn.writeGraph(base);

        ParLe p = new ParLeBayes(2.0D);

        newbn = p.go(newbn, d);

        String bn_path = base + ".uai";

        BnUaiWriter.ex(bn, bn_path);

        return newbn;
    }
}
