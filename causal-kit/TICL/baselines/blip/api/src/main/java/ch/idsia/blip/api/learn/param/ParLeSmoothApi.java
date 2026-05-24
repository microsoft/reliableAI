package ch.idsia.blip.api.learn.param;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.learn.param.ParLeSmooth;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * ParLe - Parameter Learner with Smoothing
 */

public class ParLeSmoothApi extends ParLeApi {

    private static final Logger log = Logger.getLogger(
            ParLeSmoothApi.class.getName());

    @Option(name = "-va", required = true, usage = "Validation dataset")
    protected String ph_valid;

    public static void main(String[] args) {
        defaultMain(args, new ParLeSmoothApi());
    }

    @Override
    public void exec() throws Exception {

        BayesianNetwork res = getBayesianNetwork(ph_res);
        DataSet train = getDataSet(ph_dat);

        ParLeSmooth parLe = new ParLeSmooth();

        parLe.verbose = verbose;
        parLe.thread_pool_size = thread_pool_size;

        BayesianNetwork newBn = parLe.go(res, train, ph_valid);

        writeBayesianNetwork(newBn, ph_network);
    }
}
