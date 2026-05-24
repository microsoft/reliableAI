package ch.idsia.blip.core.learn.solver.ktree;


import ch.idsia.blip.core.io.GobnilpReader;
import ch.idsia.blip.core.io.ScoreWriter;
import ch.idsia.blip.core.utils.tw.Dandelion;
import ch.idsia.blip.core.utils.tw.KTree;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.other.StreamGobbler;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * IS ALIVE! IS AAAAALIVE!
 */
public class BaseS2Solver extends ScoreSolver {

    private final Logger log = Logger.getLogger(BaseS2Solver.class.getName());

    public int tw;

    public String ph_work;

    public String ph_gobnilp;

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        this.tw = gInt("tw", 5);
    }

    @Override
    protected String name() {
        return "Gobnilp Dandelion Solver!";
    }

    @Override
    protected void prepare() {
        super.prepare();
        if (verbose > 0) {
            log("tw: " + tw + "\n");
        }

        File w = new File(ph_work);

        deleteFolder(new File(ph_work));
        w.mkdir();
    }

    @Override
    protected Searcher getSearcher() {
        return null;
    }

    @Override
    protected Provider getProvider() {
        return null;
    }

    @Override
    public BaseSearcher getNewSearcher(int i) {
        return new BaseS2Searcher(this, i);
    }

    private class BaseS2Searcher extends BaseSearcher {

        public BaseS2Searcher(BaseS2Solver solver, int i) {
            super(solver, i);
        }

        @Override
        public void run() {

            int iter = 0;

            while (still_time) {

                KTree K = sampleKtree(thread, iter);

                if (K == null) {
                    checkTime();
                    continue;
                }

                ParentSet[][] p_scores = K.selectScores(sc);

                // K.T.graph(f("%s/%d", ph_work, thread));
                String jkl = f("%s/%d-%d.jkl", ph_work, thread, iter);
                String out = f("%s/%d-%d.gob", ph_work, thread, iter);
                String err = f("%s/%d-%d.err", ph_work, thread, iter);

                ScoreWriter.go(p_scores, jkl);

                try {
                    ProcessBuilder builder = new ProcessBuilder(ph_gobnilp,
                            "-f=jkl", jkl);

                    Process p = builder.start();

                    StreamGobbler e = new StreamGobbler(p.getErrorStream(), err);

                    e.start();
                    StreamGobbler o = new StreamGobbler(p.getInputStream(), out);

                    o.start();

                    p.waitFor();

                } catch (IOException e) {
                    logExp(log, e);
                } catch (InterruptedException e) {
                    logExp(log, e);
                }

                GobnilpReader g = new GobnilpReader();

                g.go(out);

                // TODO
                // newStructure(g.new_str);

                iter++;
                checkTime();
            }

        }
    }

    KTree sampleKtree(int i, int j) {
        return KTree.decode(Dandelion.sample(n_var, tw, this));
    }

}

