package ch.idsia.blip.core.utils.tw;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.array.TIntList;
import ch.idsia.blip.core.utils.other.Pair;

import java.util.Arrays;

import static ch.idsia.blip.core.utils.data.ArrayUtils.zeros;


public class Dandelion {

    public final int[] Q;
    public final int[][] S;

    public Dandelion(int[] Q, int[][] S) {
        this.Q = Q;
        this.S = S;
    }

    /**
     * Uniformally sample a code
     */
    public static Dandelion sample(int n, int k, App b) {

        int[] Q = zeros(k);
        TIntArrayList v = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            v.add(i + 1);
        }
        for (int i = 0; i < k; i++) {
            int r = b.randInt(0, n - (i + 1));

            Q[i] = v.get(r);
            v.removeAt(r);
        }

        int[][] S = new int[2][];

        S[0] = zeros(n - k - 2);
        S[1] = zeros(n - k - 2);

        for (int i = 0; i < n - k - 2; i++) {
            int r = b.randInt(1, (n - k) * k + 1) - 1;

            if (r == (n - k) * k) {
                S[0][i] = 0;
                S[1][i] = -1;
            } else {
                S[0][i] = (r % (n - k)) + 1;
                S[1][i] = (int) (Math.floor(r / (n - k)) + 1);
            }
        }

        return new Dandelion(Q, S);
    }

    /**
     * generalized Dandelion decoding
     * program 2 in the paper
     */
    public static Pair<int[], int[]> decoding(int[][] S, int r, int x) {

        int n = S[0].length + 2;

        // construct Base from S
        TIntArrayList V = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if (i != r && i != x) {
                V.add(i);
            }
        }
        int[][] G = new int[3][];

        int v_s = V.size();
        int g_s = v_s + 1;

        for (int i = 0; i < 3; i++) {
            G[i] = new int[g_s];
        }

        for (int i = 0; i < v_s; i++) {
            G[0][i] = V.get(i);
            G[1][i] = S[0][i];
            G[2][i] = S[1][i];
        }

        G[0][v_s] = x;
        G[1][v_s] = r;
        G[2][v_s] = -1;

        int[] temp1 = Arrays.copyOf(G[0], g_s);

        Arrays.sort(temp1);
        int[] temp2 = new int[g_s];

        for (int i = 0; i < g_s; i++) {
            temp2[i] = KTree.findN(temp1[i], G[0]);
        }

        int[] p = new int[g_s];

        for (int i = 0; i < g_s; i++) {
            p[i] = G[1][temp2[i]];
        }

        int[] l = new int[g_s];

        for (int i = 0; i < g_s; i++) {
            l[i] = G[2][temp2[i]];
        }

        // identify all cycles in Base
        // processed -> 1, inProgress ->2

        int[] status = zeros(n);
        TIntArrayList mi = new TIntArrayList();

        for (int v = 0; v < n - 1; v++) {
            analyze(v, p, status, mi);
        }

        if (!mi.isEmpty()) {
            // starting from different variables, may end up in the same cycle
            int[] a = ArrayUtils.unique(mi.toArray());

            Arrays.sort(a);
            for (int m : a) {
                int tmp = l[x - 1];

                l[x - 1] = l[m - 1];
                l[m - 1] = tmp;

                tmp = p[x - 1];
                p[x - 1] = p[m - 1];
                p[m - 1] = tmp;
            }
        }

        return new Pair<int[], int[]>(p, l);
    }

    /**
     * program 3 in the paper
     * identify a cycle according to the current status
     * update the status and return the largest value in the cycle
     */
    private static void analyze(int v, int[] p, int[] status, TIntArrayList mi) {
        if (status[v] != 1) {
            status[v] = 2;
            if (p[v] != 0) {
                if (status[p[v] - 1] == 2) {
                    TIntList cycl = new TIntArrayList();

                    cycl.add(v + 1);
                    int targ = v + 1;

                    while (p[v] != targ) {
                        // there is possible that a->b->c->b, then a is not in the cycle
                        if (ismember(p[v], cycl)) {
                            int xy = cycl.indexOf(p[v]);

                            cycl = cycl.subList(xy, cycl.size());
                            break;
                        } else {
                            cycl.add(p[v]);
                            v = p[v] - 1;
                        }
                    }
                    mi.add(cycl.max());
                } else {
                    analyze(p[v] - 1, p, status, mi);
                }
                status[v] = 1;
            } else {
                status[v] = 1;
            }
        }
    }

    private static boolean ismember(int a, TIntList b) {
        return b.contains(a);
    }

    /**
     * Check a dandelion code
     */
    public void check() {

        int k = Q.length;
        int n = S[0].length + k + 2;

        if (S[0].length != S[1].length) {
            throw new Error("Different size for the two columns of S!");
        }

        // Check codeword S
        for (int i = 0; i < S[0].length; i++) {
            if (S[0][i] == 0) {
                if (S[1][i] != -1) {
                    throw new Error(
                            "Row " + i + " of S incorrect! " + S[0][i] + ", "
                            + S[1][i]);
                }
            } else if (S[1][i] == -1) {
                if (S[0][i] != 0) {
                    throw new Error(
                            "Row " + i + " of S incorrect! " + S[0][i] + ", "
                            + S[1][i]);
                }
            } else if (S[0][i] < 1 || S[0][i] > n - k) {
                throw new Error("Row " + i + " of S[0] incorrect! " + S[0][i]);
            } else if (S[1][i] < 1 || S[1][i] > k) {
                throw new Error("Row " + i + " of S[1] incorrect! " + S[1][i]);
            }
        }

    }
}
