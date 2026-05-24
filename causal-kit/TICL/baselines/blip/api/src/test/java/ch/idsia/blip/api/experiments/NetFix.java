package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import org.junit.Test;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


// Fix of .net datafile
public class NetFix extends TheTest {

    @Test
    public void new_test() throws Exception {

        String base = basePath + "netfi/";

        String data = base + "data";
        String jkl = base + "jkl";
        String str = base + "str";
        int threads = 7;

        BayesianNetwork bn = BnNetReader.ex(base + "child.net");

        BnNetWriter.ex(bn, base + "true");

        SamGe.ex(bn, data, 50000);
        DataSet dat = getDataSet(data);
        IndependenceScorer is = new IndependenceScorer();

        // is. = getScoreReader(jkl);
        is.verbose = 2;
        is.go(dat);

        AsobsSolver sv = new AsobsSolver();

        sv.verbose = 2;
        sv.init(getScoreReader(jkl), 10);
        sv.go(str);

        BayesianNetwork newBn = BnResReader.ex(str);

        BnNetWriter.ex(ParLe.ex(newBn, dat), base + "learned");

        BnNetWriter.ex(ParLe.ex(bn, dat), base + "original");

    }

}
