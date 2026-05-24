package ch.idsia.blip.api.common;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import org.kohsuke.args4j.Option;

import static ch.idsia.blip.core.utils.RandomStuff.getBayesianNetwork;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class Analyze extends Api {

    @Option(name = "-n", required = true, usage = "Bayesian network file path")
    private String ph;

    public static void main(String[] args) {
        defaultMain(args, new Analyze());
    }

    @Override
    public void exec() throws Exception {

        BayesianNetwork bn = getBayesianNetwork(ph);

        pf("Number of nodes: %d \n", bn.n_var);
        pf("Number of edges: %d \n", bn.numEdges());
        pf("Check: is acyclic? %pb \n", bn.isAcyclic());
        pf("Check treediwth: %d \n", bn.treeWidth());
    }
}
