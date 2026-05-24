package ch.idsia.blip.api.common;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.NetToGraph;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;

public class NetToGraphApi extends Api {

    private static final Logger log = Logger.getLogger(
            NetToGraphApi.class.getName());

    protected NetToGraph ntg;

    @Option(name = "-n", required = true, usage = "Bayesian network file path")
    private String path_bn;

    @Option(name = "-t", usage = "maximum execution time (seconds)")
    private int max_time;

    public NetToGraphApi() {
        ntg = new NetToGraph();
    }

    public static void main(String[] args) {
        defaultMain(args, new NetToGraphApi());
    }

    @Override
    public void exec() throws Exception {
        BayesianNetwork bn = getBayesianNetwork(path_bn);

        p(path_bn);

        String path_out = rmExt(path_bn);

        ntg.go(bn, path_out, max_time);
    }

}
