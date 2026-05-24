package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.utils.other.ExpectationMaximization;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


/**
 * Test for EM in the package
 */
public class MissingFillTest extends TheTest {

    @Test
    public void createData() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        int n_sample = 20;

        String nm_bn = "simple";
        File f_bn = new File(basePath + nm_bn + ".net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        SamGe samGe = new SamGe();

        samGe.go(bn, basePath + nm_bn, n_sample);
    }

    @Test
    // Test EM on asia data, with defined missing column
    public void testMissing() throws IOException {

        String bn = "child";

        String f_df = String.format("%s%s/%s-missing.dat", basePath, "missing",
                bn);
        DataSet dat_rd = getDataSet(f_df);

        int missingVar = 0;
        int[] set_p = new int[] { 1, 2};

        ExpectationMaximization em = new ExpectationMaximization(dat_rd);
        int[][] p = em.computeParentSetValues(set_p);

        System.out.println("p_values");
        for (int[] p1 : p) {
            System.out.println(Arrays.toString(p1));
        }

        short[] new_values = em.performEM(dat_rd, missingVar, set_p);

        String path = String.format("%s%s/%s-recover-%s.dat", basePath,
                "missing", bn, Arrays.toString(set_p));

        printValues(new_values, dat_rd, missingVar, path);

        // ExpectationMaximization em = new ExpectationMaximization(dat.n_var,
        // dat.l_n_arity);

        // fillVar(bn, dat, missingVar, em, 2);

        /* );

         for (int n = 0; n < dat.n_var; n++) {

         if (n == missingVar) {
         continue;
         }

         fillVar(bn, nm, dat, missingVar, em, n);

         }*/
    }

    /*
     private void fillVar(String bn, DataFileReader dat, int missingVar, ExpectationMaximization em, int n) throws IOException {
     TIntArrayList vars = new TIntArrayList();

     vars.add(missingVar);
     vars.add(n);

     // Do EM on the missing variable, and another variable
     int[][][] new_values = em.performEM(dat.row_values, vars.toArray());

     // copy all the dand values
     copyValues(new_values, missingVar, dat, dat.row_values);

     printValues(new_values, dat,
     String.format("%s%s/%s-recover-%d.dat", basePath, "missing", bn,
     n));
     }*/

    private void printValues(short[] new_values, DataSet dat_rd, int missingVar, String nm) throws IOException {/*
         BufferedWriter rd_writer = new BufferedWriter(
         new OutputStreamWriter(new FileOutputStream(nm), "utf-8"));

         rd_writer.write(String.format("%d\n", dat.n_var));

         rd_writer.write(
         String.format("%s\n", StringUtils.join(dat.l_nm_var, " ")));

         rd_writer.write(String.valueOf(dat.l_n_arity[0]));
         for (int i = 1; i < dat.n_var; i++) {
         rd_writer.write(" " + dat.l_n_arity[i]);
         }
         rd_writer.write("\n");

         rd_writer.write(String.format("%d\n", dat.n_datapoints));

         // Print each values
         String l;

         // For each line
         for (int i = 0; i < dat.n_datapoints; i++) {

         l = "";

         // For each variable
         for (int j = 0; j < dat.n_var; j++) {

         int set;

         if (j == missingVar) {
         set = new_values[i];
         } else {
         set = dat.sample[j][i];
         }

         if (j != 0) {
         l += " " + set;
         } else {
         l += set;
         }

         }

         l += "\n";
         rd_writer.write(l.replace("-1", "?"));
         }

         rd_writer.flush();
         */}

    private void copyValues(int[][][] new_values, int missingVar, DataSet dat_rd, int[][][] row_values) {
        for (int n = 0; n < dat_rd.n_var; n++) {

            if (n == missingVar) {
                continue;
            }

            new_values[n] = row_values[n];
        }

    }

    @Test
    public void oneAndOnlyTrueTest() throws IOException, IncorrectCallException {

        String bn = "child";
        String m = "bic";

        String path = String.format("%s%s/%s-missing", basePath, "missing", bn);
        String file = path + ".dat";
        String scores = String.format("%s-%s.scores", path, m);

        int verbose = 2;
        double max_exec_time = 2;
        int max_pset_size = 6;

        String[] args = {
            "", "-f", file, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose)};

        IndependenceScorerApi.main(args);

    }
}
