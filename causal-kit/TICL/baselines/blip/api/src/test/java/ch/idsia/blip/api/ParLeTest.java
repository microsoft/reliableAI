package ch.idsia.blip.api;


import ch.idsia.blip.api.learn.param.ParLeApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnMdlReader;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.io.dat.DatFileReader;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ParLeTest extends TheTest {

    public ParLeTest() {
        basePath = basePath + "parle/";
    }

    @Test
    public void testUstat() throws IOException, IncorrectCallException {

        String base = "/home/loskana/Documents/bazaar/2016/Ustat/U2/";

        BnResReader g = new BnResReader();
        BayesianNetwork bn = g.go(base + "inference/result");
        DataSet dat = getDataSet(base + "quarter-data.dat");
        ParLe p = new ParLeBayes(10);

        p.go(bn, dat);
        BnNetWriter.ex(p.bn, base + "inference/final");
    }

    @Test
    public void testSample() throws IOException, IncorrectCallException {
        String r = "child";
        String h = basePath + r;
        int n = 10000;

        BayesianNetwork bn = getBnFromFile(r + ".net");

        SamGe s = new SamGe();
        BnResReader g = new BnResReader();
        ParLe p = new ParLeBayes(10);

        s.go(bn, h, n);

        p.go(g.go(h + ".str"), getDataSet(h + "-" + n + ".dat"));
        BnNetWriter.ex(p.bn, getWriter(h + "-new.net"));
    }

    /* R code to test
     library("bnlearn")
     bn <- read.net("random20-3.net")
     modelstring(bn)
     */

    @Test
    public void testChild() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        String s_bn_mdl = "[BirthAsphyxia|Disease][HypDistrib|DuctFlow][HypoxiaInO2|CardiacMixing][CO2|LungParench]"
                + "[ChestXray|LungFlow][Grunting|LungParench][LVHreport|LVH][LowerBodyO2|HypoxiaInO2][RUQO2|HypoxiaInO2][CO2Report|CO2]"
                + "[XrayReport|ChestXray][Disease|CardiacMixing][GruntingReport|Grunting][Age|Disease][LVH|Disease][DuctFlow|Disease][CardiacMixing]"
                + "[LungParench|ChestXray][LungFlow|Disease][Sick|Disease]";

        testParameterLearning(s_bn_mdl, "child", 5000);
    }

    @Test
    public void testRandom20Simple() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        String s_bn_mdl = "[node3][node8][node9][node18][node0|node3][node6|node9:node18][node10|node18:node8][node15|node8]"
                + "[node17|node3][node1|node17:node9][node5|node0][node16|node6:node10][node2|node5:node8]"
                + "[node4|node5:node9:node0][node12|node1:node18][node13|node2:node0:node16][node14|node12:node16:node10]"
                + "[node11|node8:node3:node18:node13][node19|node6:node17:node14][node7|node17:node19:node3]";

        testParameterLearning(s_bn_mdl, "random20-3", 320);
    }

    @Test
    public void testRandom20Complex() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        String s_bn_mdl = "[node0][node8][node6|node8:node0][node9|node0][node15|node9:node6][node2|node15:node6]"
                + "[node10|node2:node15:node8:node0][node17|node10:node15:node0][node14|node9:node8:node10:node17:node6]"
                + "[node1|node14:node2:node10:node9][node11|node6:node1:node10:node2:node15:node17:node0]"
                + "[node3|node0:node15:node17:node14:node11:node9][node12|node6:node15:node8:node9:node3:node2]"
                + "[node13|node17:node0:node12:node9:node15:node14:node6][node4|node13:node10:node17:node3:node9:node1]"
                + "[node5|node12:node13:node2:node0:node4][node16|node9:node0:node5:node2:node10:node13:node3:node11]"
                + "[node7|node17:node16:node8:node5][node18|node9:node2:node11:node5:node12:node16:node4]"
                + "[node19|node17:node4:node7:node3:node6:node12]";

        testParameterLearning(s_bn_mdl, "random20-5", 320);

        testParameterLearning(s_bn_mdl, "random20-5", 3200);

    }

    @Test
    public void testExec() {
        exec("child-5000");
    }

    private void exec(String s) {

        String path = basePath + "scorer/" + s;

        String dat = path + ".dat";
        String str = path + ".str";

        String[] args = {
            "", "-d", dat, "-r", str};

        ParLeApi.main(args);

    }

    private void testParameterLearning(String s_bn_mdl, String nm, int n_dat) throws IOException {
        BayesianNetwork bn = BnMdlReader.go(s_bn_mdl);

        // System.out.println(bn);

        String s_bn = String.format("%s%s.net", basePath, nm);
        String s_dat = String.format("%s%s-%d.dat", basePath, nm, n_dat);

        DatFileReader dat_rd = new DatFileReader();

        dat_rd.init(s_dat);
        // Spl spl = new Spl.SplBayes(bn, dat, 1);
        ParLe parLe = new ParLeBayes(1);

        // parLe.go(res, dat);

        for (int i = 0; i < bn.n_var; i++) {
            double sum = 0;
            double[] pot = bn.potentials(i);

            for (double aPot : pot) {
                sum += aPot;
            }

            if (!doubleEquals(sum, pot.length / bn.arity(i))) {
                System.out.println(
                        String.format("%.2f - %d", sum, pot.length / bn.arity(i)));
            }
        }

        OutputStreamWriter bn_stream = new OutputStreamWriter(System.out);
        BufferedWriter bn_wr = new BufferedWriter(bn_stream);

        BnNetWriter.ex(parLe.bn, bn_wr);

        bn_wr.flush();

        DatFileLineReader dat = new DatFileLineReader(s_dat);

        BayesianNetwork bn_orig = BnNetReader.ex(s_bn);

        // System.out.println(bn);

        // System.out.println(bn_orig);

        double d = 0;

        for (int i = 0; i < dat.n_datapoints; i++) {
            short[] samp = dat.next();
            double o = bn_orig.getLogLik(samp);
            double l = bn.getLogLik(samp);

            d += Math.abs(o - l) * 100.0 / o;
        }
        System.out.println(d / dat.n_datapoints);
    }
}
