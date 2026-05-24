package ch.idsia.blip.core.common;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.io.dat.ArffFileLineWriter;
import ch.idsia.blip.core.io.dat.BaseFileLineWriter;
import ch.idsia.blip.core.io.dat.DatFileLineWriter;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.*;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getWriter;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class SamGe extends App {

    private static final Logger log = Logger.getLogger(SamGe.class.getName());

    /**
     * Network to sample from
     */
    public BayesianNetwork bn;

    /**
     * Total number of sample to generate
     */
    private int n_sample;

    /**
     * Writer for sampled datapoints (Cussen's format)
     */
    private BaseFileLineWriter wr;

    /**
     * Writer for sampled datapoints (Dataframe format)
     */
    // private DataFileWriter wr_dataframe; //

    /**
     * Flag for filling the .tab file
     */
    public boolean fill_tab;

    /**
     * Flag for missing data
     */
    public boolean missing;

    /**
     * Percentage of variables that will contain missing values
     */
    public double perc_var = 0.2;

    /**
     * Percentage of missing values
     */
    public double perc_values = 0.2;

    private TIntArrayList missing_var;

    /**
     * Start Sample Generator
     *
     * @param in_bn       Bayesian Network to work with
     * @param in_wr_path  path where to graph
     * @param in_n_sample number of sample to work with
     */
    public void go(BayesianNetwork in_bn, String in_wr_path, int in_n_sample, String format) {

        prepare();

        bn = in_bn;
        n_sample = in_n_sample;

        if (in_wr_path == null)
            return;

            Writer bf_wr = getWriter(in_wr_path);
            if (bf_wr == null)
                return;

            if (format.equals("dat")) {
                wr = new DatFileLineWriter(in_bn, bf_wr);
            } else {
                wr = new ArffFileLineWriter(in_bn, bf_wr, "new");
            }

            /*
             bf_wr = new BufferedWriter(
             new OutputStreamWriter(
             new FileOutputStream(
             String.format("%s.tab", in_wr_path)),
             "utf-8"));

             wr_dataframe = new DataFileWriter(in_bn, bf_wr, DataFileWriter.Format.Dataframe);
             */


        if (missing) {
            missing_var = new TIntArrayList();
            for (int i = 0; i < bn.n_var; i++) {
                if (Math.random() < perc_var) {
                    missing_var.add(i);
                }
            }
        }

        writeSample();

    }

    /**
     * Main execution of Sample Generator.
     */
    private void writeSample() {

        try {
            wr.writeMetaData();

            // wr_dataframe.writeMetaData(n_sample);

            int[] ord = bn.getTopologicalOrder();

            for (int i = 0; i < n_sample; i++) {

                short[] samp = getSample(ord);

                if (missing) {
                    missSample(samp);
                }

                wr.next(samp);

                if (fill_tab) {// wr_dataframe.next(thread, samp);
                }
            }
        } catch (Exception exp) {
            RandomStuff.logExp(log, exp);
        } finally {
            close();
        }
    }

    /**
     * Assign missing values under the parameters
     *
     * @param samp current sample
     */
    private void missSample(short[] samp) {
        for (int i = 0; i < missing_var.size(); i++) {
            if (randProb() < perc_values) {
                samp[missing_var.get(i)] = -1;
            }
        }
    }

    /**
     * Close the writer structures.
     */
    private void close() {
        try {
            if (wr != null) {
                wr.close();
            }
        } catch (Exception exp) {
            RandomStuff.logExp(log, exp);
        }
        try {/*
             if (wr_dataframe != null) {
             wr_dataframe.close();
             }
             */} catch (Exception exp) {
            RandomStuff.logExp(log, exp);
        }
    }

    /**
     * Obtain a sample from the bayesian network, in the given order
     *
     * @param ord topological order to visit the bayesian network nodes
     * @return a sample of bn.n_var row_values, sampled according to the bn's probabilities
     */
    public short[] getSample(int[] ord) {
        short[] sample = new short[bn.n_var];

        for (int n : ord) {
            sample[n] = getVariableSample(n, sample);
        }

        return sample;
    }

    /**
     * Get a random value for the variable, following the probability distribution determined by the given assignment of the dand variables.
     * In this assignment it is mandatory that the parents of the variable have a value.
     *
     * @param n      variable of interest
     * @param sample assignment of the dand variables
     * @return a random value for the requested variable
     */
    public short getVariableSample(int n, short[] sample) {
        int ar = bn.arity(n); // Arity for variable

        double[] pt = bn.potentials(n);
        int ix = bn.potentialIndex(n, sample);
        int j = ix * ar;

        double r = rand.nextDouble() - Math.pow(2, -10); // Get a random number in (0, 1)
        short i = 0;

        while ((i < ar) && j < pt.length && (pt[j] < r)) { // Select a value from the probabilities given the random
            r -= pt[j];
            j++;
            i++;
        }

        /*
         if (i == ar) {
         log.severe(
         String.format("thread: %s - ar: %d r: %s probs: %s", i, ar, r,
         Arrays.toString(probs)));
         }*/

        if (i >= ar) {
            i = (short) (ar - 1);
        }

        return i;
    }

    public static void ex(BayesianNetwork bn, String path, int n) {
        SamGe sg = new SamGe();

        try {
            sg.go(bn, path, n);
        } catch (Exception e) {
            logExp(log, e);
        }
    }

    public void go(BayesianNetwork bn, String path, int n) {
        go(bn, path, n, "dat");
    }
}
