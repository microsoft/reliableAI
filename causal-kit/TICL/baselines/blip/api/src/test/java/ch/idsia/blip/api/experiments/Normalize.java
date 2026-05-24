package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import org.junit.Test;

import java.io.*;
import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;


public class Normalize extends TheTest {

    @Test
    public void normalize() throws IOException {

        File folder = new File(basePath + "exp/norm/nets/");

        for (File fileEntry : folder.listFiles()) {

            String s = fileEntry.getName();

            normNet(s);
        }

    }

    private void normNet(String s) throws IOException {
        System.out.println(s);

        BufferedReader rd = new BufferedReader(
                new FileReader(basePath + "exp/norm/nets/" + s));
        BayesianNetwork bn = BnNetReader.ex(rd);

        // System.out.println(bn);

        OutputStreamWriter bn_stream = new OutputStreamWriter(
                new FileOutputStream(basePath + "exp/norm/new/" + s), "utf-8");
        BufferedWriter bn_wr = new BufferedWriter(bn_stream);

        for (int i = 0; i < bn.n_var; i++) {
            normalizePotentials(bn, i);
        }

        BnNetWriter.ex(bn, bn_wr);

        bn_stream.flush();

        bn_stream.close();
    }

    private void normalizePotentials(BayesianNetwork bn, int n) {

        int p_arity = 1;

        for (int aPar : bn.parents(n)) {
            p_arity *= bn.arity(aPar);
        }

        int ar = bn.arity(n);

        double[] probs = bn.potentials(n);

        Random rn = getRandom();

        for (int i = 0; i < p_arity; i++) {

            double sum = 0;

            for (int j = 0; j < ar; j++) {
                int k = ((i * ar) + j);

                if (probs[k] < 0.01) {
                    probs[k] = Math.abs(rn.nextInt() % 100) / 1000.0;
                }
                sum += probs[k];
            }

            for (int j = 0; j < ar; j++) {
                probs[((i * ar) + j)] /= sum;
            }
        }

        bn.l_potential_var[n] = probs;
    }
}
