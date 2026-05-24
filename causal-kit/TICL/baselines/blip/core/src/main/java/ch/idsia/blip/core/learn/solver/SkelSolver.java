package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.utils.arcs.Directed;
import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.ps.SkelProvider;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.samp.SkelSampler;
import ch.idsia.blip.core.utils.ParentSet;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public abstract class SkelSolver extends BaseSolver {

    protected Directed skel;

    protected int[][] parents;

    // Best structures found
    private List<ParentSet[]> best_str_s;

    private TreeSet<String> best_str_c;

    public void init(int max_exec, String s) {
        init(max_exec, readSkeleton(s));
    }

    public void init(int max_exec, Directed skel) {
        this.max_exec_time = max_exec;
        this.skel = skel;
        n_var = skel.n;
    }

    @Override
    protected Provider getProvider() {
        return new SkelProvider(skel);
    }

    @Override
    public Sampler getSampler() {
        return new SkelSampler(skel, this.rand);
    }

    @Override
    public void newStructure(ParentSet[] str) {
        double new_sk = getSk(str);

        synchronized (lock) {
            if (new_sk > best_sk) {
                best_sk = new_sk;
                best_str_s = new ArrayList<ParentSet[]>();
                best_str_c = new TreeSet<String>();
            }

            if (new_sk != best_sk) {
                return;
            }

            String c = getDescr(str);

            if (best_str_c.contains(c)) {
                return;
            }

            best_str_c.add(c);
            best_str_s.add(str);

            if (res_path != null) {
                cloneStr(str, best_str);
                writeGraph(
                        f("%s-%d-%d", res_path, (int) new_sk, best_str_s.size()));
            }
        }

    }

    private String getDescr(ParentSet[] str) {
        StringBuilder b = new StringBuilder();

        for (int v = 0; v < str.length; v++) {
            b.append(Arrays.toString(str[v].parents));
        }
        return b.toString();
    }

    public static Directed readSkeleton(String s) {
        try {
            BufferedReader br = new BufferedReader(getReader(s));
            int n = Integer.valueOf(br.readLine().trim());
            Directed u = new Directed(n);
            String l;

            while ((l = br.readLine()) != null) {
                String[] aux = l.split("->");

                u.mark(nt(aux, 0) - 1, nt(aux, 1) - 1);
            }
            return u;
        } catch (IOException e) {
            p(e.getMessage());
        }

        return null;
    }

    public static int nt(String[] aux, int i) {
        return Integer.valueOf(aux[i].trim());
    }
}
