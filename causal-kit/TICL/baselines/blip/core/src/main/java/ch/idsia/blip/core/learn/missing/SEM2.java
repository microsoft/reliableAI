package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.DFGenerator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.f;


public class SEM2 extends HiddenSEM2 {

    public SEM2(int verbose) {
        super(verbose);
    }

    @Override
    protected void addVars(int r, List<TIntArrayList> new_best, FastList<Integer> fs) {

        // Add new hidden variables, until each variable has at least r variables
        int cnt = 0;
        int k = 3;

        while (cnt < n_hidden && fs.size() > 1) {

            int[] f = new int[k];

            for (int i = 0; i < k; i++) {
                f[i] = fs.rand();
                while (notOk(f, i)) {
                    f[i] = fs.rand();
                }
            }

            TIntArrayList t = new TIntArrayList();

            for (int i = 0; i < k; i++) {
                new_best.get(f[i]).add(cnt + n_var);
                if (new_best.get(f[i]).size() >= k) {
                    fs.delete(f[i]);
                }
            }

            new_best.add(new TIntArrayList());
            cnt++;
        }
    }

    @Override
    public void addHiddenValues(DataSet dat, int k) throws IOException {

        int card = 3;

        String[] aux = new String[dat.n_var + k];
        int[] aux2 = new int[dat.n_var + k];
        int[][][] aux3 = new int[dat.n_var + k][][];

        DFGenerator DFg = new DFGenerator();

        for (int i = 0; i < dat.n_var; i++) {
            aux[i] = dat.l_nm_var[i];
            aux2[i] = dat.l_n_arity[i];
            aux3[i] = dat.row_values[i];
        }

        Analyzer an = new Analyzer(dat);

        for (int i = 0; i < k; i++) {
            int v = dat.n_var + i;

            aux[v] = f("MS%d", i);
            aux2[v] = card;

            TIntArrayList[] r = new TIntArrayList[card];

            for (int vv = 0; vv < card; vv++) {
                r[vv] = new TIntArrayList();
            }

            int[][] p_v = computeChildrenSetValues(v);

            for (int j = 0; j < p_v.length; j++) {
                double[] distr = DFg.generateUniformDistribution(card);

                for (int row : p_v[j]) {
                    r[DFg.sampleV(distr)].add(row);
                }
            }

            int[][] row = new int[card][];

            for (int vv = 0; vv < card; vv++) {
                row[vv] = r[vv].toArray();
                Arrays.sort(row[vv]);
            }

            aux3[v] = row;
        }

        dat.l_nm_var = aux;
        dat.l_n_arity = aux2;
        dat.row_values = aux3;

        dat.n_var += k;

        // DataFileWriter.ex(dat, path + "/em0.dat");
    }

    private int[][] computeChildrenSetValues(int c) {

        TIntArrayList aux = new TIntArrayList();

        for (int i = 0; i < n_var; i++) {
            if (find(c, bn.parents(i))) {
                aux.add(i);
            }
        }

        int[] set_p = aux.toArray();

        int p_arity = 1;

        Arrays.sort(set_p);

        for (int p : set_p) {
            p_arity *= dat.l_n_arity[p];
        }

        int[][] p_values = new int[p_arity][];

        for (int i = 0; i < p_arity; i++) {
            int[] values = null;

            int n = i;

            // Get value for the parent in this configuration
            for (int p : set_p) {
                int ar = dat.l_n_arity[p];
                short val = (short) (n % ar);

                n /= ar;

                // System.out.print("p: " + p + " v: " + val + ", ");

                // Update set containing sample rows for the chosen configuration
                // System.out.printf("%d (%d) - %d - %d\n", p,  dat.l_n_arity[p], val, dat.row_values[p].length);
                int[] par_var = dat.row_values[p][val];

                if (values == null) {
                    values = par_var;
                } else {
                    values = ArrayUtils.intersect(values, par_var);
                }
            }

            p_values[i] = values;
        }

        return p_values;
    }

}
