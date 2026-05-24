package ch.idsia.blip.api;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.common.LLEval;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.p;
import static org.junit.Assert.assertTrue;


public class LLEvalTest extends TheTest {

    @Test
    public void testSimple() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        File f_bn = new File(basePath + "old/random10-1.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        short[] samp2 = { 0, 1, 1, 0, 0, 1, 0, 0, 1, 0};
        double tot = 0.0;

        for (int i = 0; i < bn.n_var; i++) {
            double p = bn.getPotential(i, samp2);

            // System.out.println(thread);
            // System.out.println(p);
            // System.out.println(Math.log(p));
            tot += Math.log(p);
        }
        System.out.println(tot);
        assertTrue(Math.abs(-17.25599624476364 - tot) < Math.pow(2, -15));
    }

    @Test
    public void testAdvanced() throws IOException {
        File f_bn = new File(basePath + "old/random10-1.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        String rd2 = basePath + "old/random10-1-100.dat";

        DatFileLineReader df = new DatFileLineReader(rd2);

        df.readMetaData();

        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -7.136878717146114));
        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -3.9740971631149304));
        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -5.979141747453868));

        String f_df = basePath + "old/random10-1-100.dat";

        df = new DatFileLineReader(f_df);

        // short samp[] = df.next();
        // System.out.println(Arrays.toString(samp));

        LLEval LLEval = new LLEval();

        LLEval.go(bn, df);

        assertTrue(RandomStuff.doubleEquals(LLEval.ll, -6.1740487694943065));

    }

    @Test
    public void testSuper3() throws IOException {
        File f_bn = new File(basePath + "scorer/simple.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        String f_df = basePath + "scorer/simple-5000.dat";
        DatFileLineReader df = new DatFileLineReader(f_df);

        LLEval LLEval = new LLEval();

        LLEval.go(bn, df);

        p("\n\n");

        df = new DatFileLineReader(f_df);
        LLEval.go(bn, df);
    }

    @Test
    public void testSuper() throws IOException {
        File f_bn = new File(basePath + "old/child.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        String f_df = basePath + "scorer/child-50000.dat";
        DatFileLineReader df = new DatFileLineReader(f_df);

        LLEval LLEval = new LLEval();

        LLEval.go(bn, df);

        assertTrue(RandomStuff.doubleEquals(LLEval.ll, -18.468260242221746));

        f_bn = new File(basePath + "old/child-1.net");
        rd = new BufferedReader(new FileReader(f_bn));
        bn = BnNetReader.ex(rd);

        f_df = basePath + "scorer/child-50000.dat";
        df = new DatFileLineReader(f_df);

        df.readMetaData();

        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -11.089654331442057));
        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -10.267788616596789));

        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -8.371878556659698));

        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -8.57480446786445));

        assertTrue(
                RandomStuff.doubleEquals(bn.getLogLik(df.next()),
                -9.676357438407592));

        df = new DatFileLineReader(f_df);
        LLEval = new LLEval();
        LLEval.go(bn, df);

    }

    @Test
    public void testSuper2() throws IOException {

        File f_bn = new File(basePath + "old/child-1.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        short[] samp2 = {
            0, 1, 0, 0, 2, 0, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 1, 0};
        double tot = 0.0;

        for (int i = 0; i < bn.n_var; i++) {
            double p = bn.getPotential(i, samp2);

            // System.out.println(thread);
            // System.out.println(p);
            // System.out.println(Math.log(p));
            tot += Math.log(p);
        }
    }

    /* R code to test
     library("bnlearn")
     bn <- read.net("random10-1.net")
     fn <- read.table("random10-1-check.tab")
     logLik(bn, fn)
     logLik(bn, fn[1, ])
     logLik(bn, fn[2, ])
     logLik(bn, fn[3, ])
     */
}
