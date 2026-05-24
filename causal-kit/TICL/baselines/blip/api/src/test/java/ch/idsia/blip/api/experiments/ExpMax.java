package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.samp.SimpleSampler;
import ch.idsia.blip.core.learn.solver.src.asobs.AsobsSearcher;
import ch.idsia.blip.core.learn.solver.src.obs.ObsGreedySearcher;
import ch.idsia.blip.core.learn.solver.src.obs.ObsSearcher;
import org.junit.Test;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.getRandom;
import static ch.idsia.blip.core.utils.RandomStuff.pf;
import static org.junit.Assert.assertTrue;


public class ExpMax extends TheTest {

    String results = basePath + "asobs/opt/";

    @Test
    public void whaaaaaaaaat() throws Exception {

        for (String d : new String[] { "net50.c", "net100.c"}) {
            whaaaatGO(d);
        }

    }

    private void whaaaatGO(String d) throws Exception {

        String jkl = f("%s/%s.jkl", results, d);
        ScoreReader s = new ScoreReader(jkl);

        s.readScores();

        AsobsSolver sv = new AsobsSolver();

        sv.verbose = 2;
        sv.max_exec_time = Integer.MAX_VALUE;
        sv.prepare();

        SimpleSampler samp = new SimpleSampler(s.n_var, getRandom());

        samp.init();

        ObsSearcher obsSe = new ObsSearcher(sv);

        obsSe.init(s.m_scores, 0);
        AsobsSearcher asobsSe = new AsobsSearcher(sv);

        asobsSe.init(s.m_scores);
        ObsGreedySearcher obsGSe = new ObsGreedySearcher(sv);

        obsGSe.init(s.m_scores);
        AsobsSearcher asobsGSe = new AsobsSearcher(sv);

        asobsGSe.init(s.m_scores);

        for (int i = 0; i < 10; i++) {
            int[] ord = samp.sample();

            obsSe.search();
            double obs = obsSe.sk;

            asobsSe.search();
            double asobs = asobsSe.sk;

            obsGSe.search();
            double grobs = obsGSe.sk;

            asobsGSe.search();
            double grasobs = asobsGSe.sk;

            pf("%10s -> %10.2f , %10s -> %10.2f \n", "obs", obs, "asobs", asobs);
            pf("%10s -> %10.2f , %10s -> %10.2f \n\n", "gr-obs", grobs,
                    "gr-asobs", grasobs);

            assertTrue(asobs > obs);
            assertTrue(grobs >= obs);
            assertTrue(grasobs >= asobs);

        }

    }

}
