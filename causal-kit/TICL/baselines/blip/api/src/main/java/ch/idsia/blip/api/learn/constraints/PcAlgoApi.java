package ch.idsia.blip.api.learn.constraints;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.learn.constraints.PcAlgo;
import org.kohsuke.args4j.Option;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class PcAlgoApi extends Api {

    @Option(name = "-d", required = true, usage = "Datafile path (.dat format)")
    private static String ph_dat;

    @Option(name = "-n", required = true, usage = "Bayesian network file path")
    private static String ph_net;

    private final PcAlgo pc;

    public PcAlgoApi() {
        pc = new PcAlgo();
    }

    protected static void main(String[] args) {
        defaultMain(args, new PcAlgoApi());
    }

    @Override
    public void exec() throws Exception {
        DataSet dat = getDataSet(ph_dat);

        pc.verbose = verbose;
        pc.execute(dat);
    }

}
