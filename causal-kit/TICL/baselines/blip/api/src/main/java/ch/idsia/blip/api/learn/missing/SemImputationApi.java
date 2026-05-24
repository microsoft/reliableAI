package ch.idsia.blip.api.learn.missing;

import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.missing.HardMissingSEM;
import ch.idsia.blip.core.utils.RandomStuff;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.p;

public class SemImputationApi extends Api {

    private static final Logger log = Logger.getLogger(SemImputationApi.class.getName());

    @Option(name = "-d", required = true, usage = "Datafile input path")
    public String ph_dat;

    @Option(name = "-o", required = true, usage = "Datafile output path")
    public String ph_imputed;

    @Option(name = "-r", required = true, usage = "Bayesian network output path")
    public String ph_bn;

    @Option(name = "-tmp", required = true, usage = "Temporary directory")
    public String temp_dir;

    @Option(name = "-t", usage = "Time regulation parameter")
    public int time = 1;

    @Option(name = "-w", usage = "Learning treewidth")
    public int treewidth = 6;

    public static void main(String[] args) {
        defaultMain(args, new SemImputationApi());
    }

    private HardMissingSEM mSem;

    public void exec() throws Exception {

        mSem = new HardMissingSEM();

        mSem.init(this.temp_dir, this.time, this.treewidth);
        mSem.thread_pool_size = this.thread_pool_size;

        BayesianNetwork bn = mSem.go(ph_dat);

        RandomStuff.writeBayesianNetwork(bn, ph_bn);

        File fi = new File(ph_imputed);
        if (!fi.exists()) {
            mSem.writeCompletedDataSet(ph_dat, bn, ph_imputed);
        }
    }
}
