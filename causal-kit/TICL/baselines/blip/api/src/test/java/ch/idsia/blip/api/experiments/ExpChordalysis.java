package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import org.junit.Test;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class ExpChordalysis extends TheTest {

    String path = "/home/loskana/Desktop/chordalysis/";

    @Test
    public void whaaaaaaaaat() throws Exception {

        File f = new File(path + "result/");

        for (File file : f.listFiles()) {
            if (!file.isDirectory()) {
                continue;
            }

            String b = file.getAbsolutePath() + "/res";
            BayesianNetwork bn = BnResReader.ex(b);

            String d = String.format("%s/data/%s.dat", path, file.getName());
            DataSet dat = getDataSet(d);

            p(b);

            BayesianNetwork bn2 = ParLeBayes.ex(bn, dat);

            BnUaiWriter.ex(String.format("%s/nets/%s.uai", path, file.getName()),
                    bn2);
        }
    }
}
