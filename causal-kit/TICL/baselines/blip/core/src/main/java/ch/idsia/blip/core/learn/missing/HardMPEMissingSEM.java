package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class HardMPEMissingSEM extends HardMissingSEM {

    protected Runnable getEmSearcher(int t) {
        return new HardEmJointSearcher(t);
    }

    private class HardEmJointSearcher extends  HardEmSearcher {

        private final String ev;
        private final String qu;
        private final String ot;

        public HardEmJointSearcher(int t) {
            super(t);
            ev = f("%s/ev-%d", path, t);
            qu = f("%s/qu-%d", path, t);
            ot = f("%s/ot-%d", path, t);
        }

        @Override
        protected void expect(VariableElimination vEl, int r, TIntArrayList q, TIntIntHashMap e) {

            String cmd = f("./daoopt -f %s -e %s", bnPath, ev);

            Process proc = null;
            try {

               //  writeQuery(qu, q);
                writeEvidence(ev, e);

                proc = Runtime.getRuntime().exec(cmd, new String[0],
                        new File(System.getProperty("user.home") + "/Tools"));

                TIntIntHashMap res = getOutput(RandomStuff.exec(proc), q);

                proc.getInputStream().close();
                proc.getOutputStream().close();
                proc.getErrorStream().close();

                addResult(r, res);

            } catch (IOException exp) {
               logExp(exp);
            } catch (InterruptedException exp) {
                logExp(exp);
            }

        }


    }

}
