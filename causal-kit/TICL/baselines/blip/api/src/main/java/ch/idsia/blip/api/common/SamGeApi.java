package ch.idsia.blip.api.common;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.kohsuke.args4j.Option;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.getBayesianNetwork;
import static ch.idsia.blip.core.utils.RandomStuff.getWriter;
import static ch.idsia.blip.core.utils.RandomStuff.p;


/**
 * Sample Generator
 */

public class SamGeApi extends Api {

    @Option(name = "-n", required = true, usage = "Bayesian network file path")
    private String s_bn;

    @Option(name = "-d", required = true, usage = "Datafile path (.dat format)")
    private String s_datafile;

    @Option(name = "-set", usage = "Number of samples")
    private Integer n_sample = 10000;

    @Option(name = "-f", usage = "Output format (dat, arff)")
    private String format = "dat";

    /**
     * Command line execution
     *
     * @param args parameters provided
     */
    public static void main(String[] args) {
        defaultMain(args, new SamGeApi());
    }

    public void exec() throws Exception {
        BayesianNetwork bn = getBayesianNetwork(s_bn);

        if (bn == null) {
            return;
        }
        SamGe samGe = new SamGe();

        samGe.seed = seed;
        samGe.go(bn, s_datafile, n_sample, format);
    }

    @Override
    protected void check() throws IncorrectCallException {
        if ( ! new File(s_bn).exists()) {
            throw new IncorrectCallException( "Bayesian network input file does not exists.");
        }

        if ( getWriter(s_datafile) == null) {
            throw new IncorrectCallException( "Can't write to data result file.");
        }

    }

    /*
     @Option(name="m", samGe.missing, false, false,
     "missing values options");

     @Option(name="v", samGe.perc_var, 0.2, false,
     "Percentage of variables that will contain missing values");

     @Option(name="l", samGe.perc_values, 0.2, false,
     "Percentage of missing values");
     } */
}
