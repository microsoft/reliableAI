package ch.idsia.blip.api.old;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.utils.other.KLDiv;
import ch.idsia.blip.core.utils.other.IncorrectCallException;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.logging.Logger;


public class KlDivApi extends Api {

    private static final Logger log = Logger.getLogger(KlDivApi.class.getName());

    private final KLDiv klDiv;

    String ph_P;

    String ph_Q;

    Boolean notlogComp;

    public KlDivApi() {
        klDiv = new KLDiv();
    }

    public static void main(String[] args) throws IncorrectCallException {
        defaultMain(args, new KlDivApi());
    }

    /*

     @Override
     public void defineOpts(Options o) {
     @Option(name="t", ph_P, null, true, "Path to true network (in .net format)");
     @Option(name="q", ph_Q, null, true, "Path to learned network (in .net format)");
     @Option(name="v", klDiv.verbose, false, false, "verbose");
     @Option(name="l", notlogComp, false, false, "disable log computation");
     }
     */

    @Override
    public void exec() throws Exception {

        // Parameters validation
        if (ph_P == null) {
            throw new IncorrectCallException(
                    "No valid path to true network provided");
        }
        File f_P = new File(ph_P);

        if (ph_Q == null) {
            throw new IncorrectCallException(
                    "No valid path to Q network provided");
        }
        File f_Q = new File(ph_Q);

        BufferedReader rd_P = null;
        BufferedReader rd_Q = null;

        BayesianFactor.logComp = !notlogComp;

        rd_P = new BufferedReader(new FileReader(f_P));
        BayesianNetwork bn_P = BnNetReader.ex(rd_P);

        rd_Q = new BufferedReader(new FileReader(f_Q));
        BayesianNetwork bn_Q = BnNetReader.ex(rd_Q);

        double kl = klDiv.getKLDivergence(bn_P, bn_Q);

        System.out.println(kl);

    }
}
