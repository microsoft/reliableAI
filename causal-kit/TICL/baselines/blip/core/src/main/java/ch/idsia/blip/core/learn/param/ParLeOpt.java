package ch.idsia.blip.core.learn.param;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.LLEval;

import java.io.FileNotFoundException;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSetReader;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class ParLeOpt extends ParLeSmooth {


    public BayesianNetwork go(BayesianNetwork res, DataSet train, String valid) throws FileNotFoundException {

        prepare(res, train, valid);

        double low_find = -8;
        double high_find = 8;

        for (int i = 0; i < 10; i++) {
            double mid = (low_find + high_find) / 2;
            double mid1 = (low_find + mid) / 2;
            double mid2 = (high_find + mid) / 2;

            double mid1_ll = goParle(Math.pow(2, mid1));
            double mid2_ll = goParle(Math.pow(2, mid2));

            if (mid1_ll < mid2_ll) {
                low_find = mid;
            } else {
                high_find = mid;
            }
        }

 //       pf("%.2f %.2f \n", highestLL, highestA);

        return best;
    }

    private double goParle(double alpha) {

        ParLeBayes p = new ParLeBayes(alpha);
        BayesianNetwork newBn = p.go(res, train);
        LLEval l = new LLEval();

        l.go(newBn, getDataSetReader(valid));
        propose(l.ll, newBn, alpha);
        return l.ll;
    }


    public static BayesianNetwork ex(BayesianNetwork res, DataSet train, String valid) throws FileNotFoundException {
        return new ParLeOpt().go(res, train, valid);
    }
}
