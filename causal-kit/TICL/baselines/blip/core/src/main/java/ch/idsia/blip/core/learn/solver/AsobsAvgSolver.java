package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.ParentSet;

import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.zeros;
import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * Maj Rule - because why not?
 */
public class AsobsAvgSolver extends AsobsSolver {

    private Logger log = Logger.getLogger(AsobsAvgSolver.class.getName());

    private Map<Integer, Double> mtx;

    private double tot;

    public String ph_dat;

    private DataSet dat;

    @Override
    protected String name() {
        return "Maj Rule";
    }

    @Override
    public void prepare() {
        super.prepare();

        dat = getDataSet(ph_dat);

        this.mtx = new HashMap<Integer, Double>();
        this.tot = 0;
    }

    @Override
    protected void writeStructure(String s) {
        try {
            Writer w = getWriter(s);

            wf(w, "%d %d \n", n_var, mtx.size());

            mtx = sortByValues(mtx);

            for (Integer i : mtx.keySet()) {
                int[] arr = r_index(i, sc.length);

                if (arr[0] == arr[1]) {
                    Undirected n = new Undirected(sc.length);
                    int i2 = n.index(arr[0], arr[1]);

                    p("DOUBLE ERRORE!");
                }
                if (arr[0] >= n_var || arr[1] >= n_var) {
                    p("ERROR");
                }
                double v = mtx.get(i);

                v /= tot;
                wf(w, "%d,%s,%d,%s,%.6f\n", arr[0], n(arr[0]), arr[1], n(arr[1]),
                        Math.abs(v));
            }
            w.close();

        } catch (IOException e) {
            logExp(log, e);
        }
    }

    private String n(int i) {

        if (i >= n_var) {
            p("ERROR");
        }
        return dat.l_nm_var[i];
    }

    public int[] r_index(int i, int n) {
        int[] arr = zeros(2);

        n -= 1;

        while (i >= n) {
            i -= n;
            n--;
            arr[0]++;
        }
        arr[1] = arr[0] + i + 1;

        return arr;
    }

    @Override
    public void newStructure(ParentSet[] new_str) {
        // super.newStructure(new_sk, new_str);


        atLeastOne = true;

        double new_sk = getSk(new_str);

        // pf("New map! score %.2f \n", new_sk);

        synchronized (lock) {
            BayesianNetwork bn = new BayesianNetwork(new_str);
            Undirected u = bn.moralize();

            tot += new_sk;
            for (int i = 0; i < u.size; i++) {
                if (u.arcs[i]) {
                    if (!mtx.containsKey(i)) {
                        mtx.put(i, 0.0);
                    }
                    mtx.put(i, mtx.get(i) + new_sk);
                }
            }
        }

    }
}
