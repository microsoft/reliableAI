package ch.idsia.blip.api.old;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.utils.other.DivConq;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.ParentSet;
import org.kohsuke.args4j.Option;

import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * Divide et Conquera
 */

public class DivConqApi extends Api {

    private static final Logger log = Logger.getLogger(
            DivConqApi.class.getName());

    private final DivConq dv;

    @Option(name = "-d", required = true, usage = "Datafile path (.dat format)")
    private String ph_dat;

    private String ph_cluster;

    private String ph_scores;

    public DivConqApi() {
        dv = new DivConq();
    }

    public static void main(String[] args) throws IncorrectCallException {
        defaultMain(args, new DivConqApi());
    }

    private void writeSubScores(ScoreReader sc, String sub_problem, int[] vars) {
        Writer w = null;

        Arrays.sort(vars);
        try {
            w = getWriter(sub_problem);
            for (int v : vars) {
                List<ParentSet> toWrite = new ArrayList<ParentSet>();

                for (ParentSet pset : sc.m_scores[v]) {
                    if (isContained(pset, vars)) {
                        toWrite.add(pset);
                    }
                }

                w.write(String.format("%d ", v));
                w.write(String.format("%d\n", toWrite.size()));
                for (ParentSet pset : toWrite) {
                    w.write(String.format("%s\n", pset.prettyPrint()));
                }
                w.flush();
            }
        } catch (Exception ex) {
            logExp(log, ex);
        } finally {
            closeIt(log, w);
        }
    }

    private boolean isContained(ParentSet pset, int[] vars) {
        for (int p : pset.parents) {
            if (Arrays.binarySearch(vars, p) < 0) {
                return false;
            }
        }
        return true;
    }

    /*
     @Override
     public void defineOpts(Options o) {
     @Option(name="f", ph_dat, null, true,
     "datapoints input file");

     @Option(name="set", ph_scores, null, true,
     "scores input file");

     @Option(name="o", ph_cluster, null, true,
     "output file");

     @Option(name="tw", dv.k, 5, false,
     "number of clusters");

     @Option(name="v", dv.verbose, 0, false,
     "verbose level (1: performance, 2: conclude, 3: eval order)");
     }*/

    @Override
    public void exec() throws Exception {
        DataSet dat = getDataSet(ph_dat);

        checkPath(ph_cluster);

        dv.findKMedoids(dat);
    }
}
