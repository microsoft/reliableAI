package ch.idsia.blip.api.learn.param;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * ParLe - Parameter Learner
 */

public class ParLeApi extends Api {

    private static final Logger log = Logger.getLogger(ParLeApi.class.getName());

    @Option(name = "-r", required = true, usage = "Network map")
    protected String ph_res;

    @Option(name = "-d", required = true, usage = "Training dataset")
    protected String ph_dat;

    @Option(name = "-n", required = true, usage = "Output path")
    protected String ph_network;

    @Option(name = "-a", usage = "Equivalent sample size")
    protected double d_alpha = 1.0;

    @Option(name = "-e", usage = "Epsilon")
    protected double epsilon = 0.5;

    @Option(name = "-t", usage = "Method for learning")
    protected String s_method = "bayes";

    public static void main(String[] args) {
        defaultMain(args, new ParLeApi());
    }

    /**
     * Get the correct value for the algorithm to use from pa description.
     *
     * @param t description of the method
     * @return correct method enum
     */
    private static Method getMethodValueOf(String t) {
        if ("bayes".equals(t)) {
            return Method.Bayes;
        } else if ("mle".equals(t)) {
            return Method.Mle;
        } else if ("weight".equals(t)) {
            return Method.Ent;
        } else if ("avg".equals(t)) {
            return Method.Avg;
        } else if ("min".equals(t)) {
            return Method.Min;
        } else if ("cano".equals(t)) {
            return Method.Cano;
        }

        log.severe("No valid method selected!");
        return null;
    }

    @Override
    public void exec() throws Exception {

        BayesianNetwork res = getBayesianNetwork(ph_res);
        DataSet dat_rd = getDataSet(ph_dat);

        ParLe parLe = new ParLeBayes(d_alpha);

        parLe.verbose = verbose;

        /*
         Method method = getMethodValueOf(s_method);

         if (method == Method.Ent) {
         parLe = new ParLe.ParLeEnt(d_alpha, epsilon);
         } else if (method == Method.Mle) {
         parLe = new ParLe.ParLeMle();
         } else if (method == Method.Avg) {
         parLe = new ParLe.ParLeAvg(5);
         } else if (method == Method.Cano) {
         parLe = new ParLe.ParLeCano(5);
         } else {
         parLe = new ParLeBayes(d_alpha);
         } */

        BayesianNetwork newBn = parLe.go(res, dat_rd);

        writeBayesianNetwork(newBn, ph_network);
    }

    public enum Method {
        Bayes, Ent, Mle, Avg, Min, Cano
    }

}
