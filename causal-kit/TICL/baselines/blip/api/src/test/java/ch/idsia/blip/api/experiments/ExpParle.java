package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.*;
import ch.idsia.blip.core.learn.param.ParLe;
import org.junit.Test;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class ExpParle extends TheTest {

    public ExpParle() {
        basePath += "/parle/";
    }

    @Test
    public void gen() throws IOException, InterruptedException {
        int n = 50000;
        String h = basePath + "child";
        BayesianNetwork bn = BnNetReader.ex(h + ".net");

        SamGe.ex(bn, h, n);
        DataSet dat = getDataSet(f("%s-%d.dat", h, n));
        BayesianNetwork bnNew = ParLe.ex(bn, dat);

        BnNetWriter.ex(bnNew, h + "-new");

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec(f("meld %s.net %s-new.net", h, h));

        pr.waitFor();
    }

    @Test
    public void gen2() throws IOException, InterruptedException {
        int n = 100000;
        String h = basePath + "BN_28";
        BayesianNetwork bn = BnErgReader.ex(h + ".erg");
        String d = f("%s-%d.dat", h, n);

        SamGe.ex(bn, d, n);
        DataSet dat = getDataSet(d);
        BayesianNetwork bnNew = ParLe.ex(bn, dat);

        BnErgWriter.ex(h + "-new", bnNew);

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec(f("meld %s.erg %s-new.erg", h, h));

        pr.waitFor();
    }

    @Test
    public void gen3() throws IOException, InterruptedException {
        int n = 1000000;
        String h = basePath + "BN_28";
        BayesianNetwork bn = BnUaiReader.ex(h + ".uai");

        SamGe.ex(bn, h, n);
        DataSet dat = getDataSet(f("%s-%d.dat", h, n));
        BayesianNetwork bnNew = ParLe.ex(bn, dat);

        BnUaiWriter.ex(h + "-new", bnNew);

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec(f("meld %s.uai %s-new.uai", h, h));

        pr.waitFor();
    }

}
