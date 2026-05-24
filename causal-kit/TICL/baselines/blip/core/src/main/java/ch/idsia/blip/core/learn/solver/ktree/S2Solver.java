package ch.idsia.blip.core.learn.solver.ktree;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.tw.KTree;
import ch.idsia.blip.core.utils.tw.KTreeSampler;
import ch.idsia.blip.core.utils.other.IncorrectCallException;

import java.io.IOException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


/**
 * IS ALIVE! IS AAAAALIVE!
 */
public class S2Solver extends BaseS2Solver {

    private static final Logger log = Logger.getLogger(S2Solver.class.getName());

    public String ph_dat;

    public MutualInformation mi;

    protected KTreeSampler sampler;

    private final Object lock = new Object();

    @Override
    protected void prepare() {
        super.prepare();

        if (mi == null) {
            prepareMI();
        }

    }

    @Override
    protected void almost() {
        super.almost();

        sampler = new KTreeSampler(n_var, tw, mi, sc, this);
    }

    private void prepareMI() {

        try {
            if (ph_dat == null) {
                throw new IncorrectCallException("No datafile provided!");
            }
            if (verbose > 0) {
                log("ph_dat: " + ph_dat + "\n");
            }
            DataSet dat = getDataSet(ph_dat);

            mi = new MutualInformation(dat);
            if (verbose > 0) {
                log("Computing mi... \n");
            }
            mi.compute();
        } catch (IncorrectCallException e) {
            logExp(log, e);
        } catch (IOException e) {
            logExp(log, e);
        }
    }

    @Override
    protected String name() {
        return "Gobnilp Dandelion Solver!";
    }

    @Override
    protected KTree sampleKtree(int i, int j) {
        synchronized (lock) {
            return sampler.go();
        }
    }
}
