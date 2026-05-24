package ch.idsia.blip.core.utils.tw;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.arcs.Directed;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.other.ValueIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * KTREE_DECODING generate the graph of a tw-tree from a Dandelion code
 * <p/>
 * implementation based on the work:
 * Caminiti, Fusco, Retreschi, Bijective Linear Time Coding and
 * Decoding for tw-trees. Theory Comput Syst, 46:2, pp. 284-300, 2010.
 * <p/>
 * Input:
 * Q, a node set containing the root nodes
 * S, codeword, an m by 2 matrix whose rows are either [0 -1] or [thread j],
 * where thread is an integer in [1, n-tw] and tw is an integer in [1,tw]
 * <p/>
 * Output: a tw-tree T_k
 * represent in n*n matrix T_k,
 * T_k(thread, j) = T_k(j, thread) = 1, if there is a link between thread and j
 * T_l(thread, j) = 0, otherwise
 */

public class KTree {

    public final Undirected T;

    public KTree(Undirected T) {
        this.T = T;
    }

    public static KTree decode(Dandelion D) {

        int[] Q = D.Q;
        int[][] S = D.S;

        Arrays.sort(Q);
        int k = Q.length;
        int n = S[0].length + k + 2;

        D.check();

        // from code A_k^n to B_k^n, q_bar = min(setdiff(1:n, Q));
        int q_bar = 0;

        for (int i = 1; i < n && q_bar < 1; i++) {
            if (!ArrayUtils.find(i, Q)) {
                q_bar = i;
            }
        }

        // a permutation of variables, transfer a general tw-tree into a Renyi tw-tree
        int[] phi = compute_phi(Q, n, k);

        // all leaves of T, T is the characteristic tree of T_k
        int[] un = ArrayUtils.unique(S[0]);

        Arrays.sort(un);
        TIntArrayList L = new TIntArrayList();

        for (int i = 1; i < n - k; i++) {
            if (!ArrayUtils.find(i, un)) {
                L.add(i);
            }
        }

        int[] k_leaves = new int[L.size()];

        for (int i = 0; i < L.size(); i++) {
            k_leaves[i] = findN(L.get(i), phi) + 1;
        }

        int l_m = ArrayUtils.max(k_leaves);

        // need to insert (0, -1) corresponding to phi(l_m)
        TIntArrayList a = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if (i != phi[q_bar - 1]) {
                a.add(i);
            }
        }
        int pos = a.indexOf(phi[l_m - 1]);

        int[] s_0_n = new int[S[0].length + 1];

        System.arraycopy(S[0], 0, s_0_n, 0, pos - 1);
        s_0_n[pos - 1] = 0;
        System.arraycopy(S[0], pos - 1, s_0_n, pos, S[0].length - (pos - 1));
        S[0] = s_0_n;

        int[] s_1_n = new int[S[1].length + 1];

        System.arraycopy(S[1], 0, s_1_n, 0, pos - 1);
        s_1_n[pos - 1] = -1;
        System.arraycopy(S[1], pos - 1, s_1_n, pos, S[1].length - (pos - 1));
        S[1] = s_1_n;

        // generalized Dandelion decoding
        Pair<int[], int[]> res = Dandelion.decoding(S, 0, phi[q_bar - 1]);
        int[] p = res.getFirst();
        int[] l = res.getSecond();

        // find the breadth first order
        int n_T = p.length;
        int[] a1 = new int[n_T * 2];
        int[] a2 = new int[n_T * 2];

        for (int v = 0; v < n_T; v++) {
            a1[v] = v + 2;
            a1[v + n_T] = p[v] + 1;
            a2[v + n_T] = v + 2;
            a2[v] = p[v] + 1;
        }
        // A = sparse([2:n_T+1, p'+1],[p'+1, 2:n_T+1], 1, n_T+1, n_T+1);
        Directed A = new Directed(n_T + 2);

        for (int v = 0; v < n_T * 2; v++) {
            A.mark(a1[v], a2[v]);
            // System.out.println(a1[v] + " " + a2[v]);
        }
        // [d dt pred] = bfs(A,1);
        int[] d = bfs(A, 1);
        // [x, y] = sort(d);
        Pair<int[], int[]> g = sort(d);
        int[] x = g.getFirst();
        int[] y = g.getSecond();

        // rebuild the Renyi tw-tree
        // initialized R_k as the tw-clique, program 8 in the paper
        Undirected R_k = new Undirected(n);

        for (int i = n - k; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                R_k.mark(i, j);
            }
        }

        // R = n-tw+1:n;
        int[] R = new int[k];

        for (int i = 0; i < k; i++) {
            R[i] = n - k + i + 1;
        }
        int[][] K = new int[n][];

        // K = zeros(n, tw);
        for (int i = 0; i < n; i++) {
            K[i] = ArrayUtils.zeros(k);
        }
        // K(1, :) = R;
        ArrayUtils.cloneArray(R, K[0]);
        // for thread = 2:n_T+1
        for (int i = 1; i < n_T + 1; i++) {
            int v = y[i] - 1;

            if (p[v] == 0) {
                ArrayUtils.cloneArray(R, K[v]);
            } else {
                // int w = K[p[v]][l[v]];
                int[] temp = new int[k];

                ArrayUtils.cloneArray(K[p[v] - 1], temp);
                temp[l[v] - 1] = p[v];
                Arrays.sort(temp);
                ArrayUtils.cloneArray(temp, K[v]);
            }

            for (int K_v : K[v]) {
                R_k.mark(v, K_v - 1);
            }
        }

        /*
         for (int j = 0; j < R_k.size; j++)
         if (R_k.check(j))
         System.out.println(Arrays.toString(R_k.r_index(j)));
         */

        Undirected T_k = new Undirected(n);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (R_k.check(phi[i] - 1, phi[j] - 1)) {
                    T_k.mark(i, j);
                }
            }
        }

        return new KTree(T_k);
    }

    private static Pair<int[], int[]> sort(int[] d) {
        List<ValueIndex> a = new ArrayList<ValueIndex>();

        for (int i = 0; i < d.length; ++i) {
            a.add(new ValueIndex(d[i], i));
        }
        Collections.sort(a);
        TIntArrayList b = new TIntArrayList();
        TIntArrayList c = new TIntArrayList();

        for (ValueIndex e : a) {
            b.add(e.value);
            c.add(e.index);
        }
        return new Pair<int[], int[]>(b.toArray(), c.toArray());
    }

    // BFS Compute breadth first search distances, times, and tree for a graph
    private static int[] bfs(Directed A, int u) {
        Pair<int[], int[]> r = sparse_to_csr(A);
        int[] rp = r.getFirst();
        int[] ci = r.getSecond();

        int n = rp.length - 1;
        int[] d = ArrayUtils.arr(n, -1);
        int[] dt = ArrayUtils.arr(n, -1);

        int[] pred = ArrayUtils.zeros(n);
        int[] sq = ArrayUtils.zeros(n);

        int sqt = 0;
        int sqh = 0;

        sqt = sqt + 1;
        sq[sqt - 1] = u;
        int t = 0;

        d[u - 1] = 0;
        dt[u - 1] = t;
        t++;
        pred[u - 1] = u;

        while (sqt - sqh > 0) {
            sqh = sqh + 1;
            int v = sq[sqh - 1];

            // pop v off the head of the queue
            for (int ri = rp[v - 1]; ri < rp[v]; ri++) {
                int w = ci[ri - 1];

                if (d[w - 1] < 0) {
                    sqt = sqt + 1;
                    sq[sqt - 1] = w;
                    d[w - 1] = d[v - 1] + 1;
                    dt[w - 1] = t;
                    t = t + 1;
                    pred[w - 1] = v;
                }

            }
        }
        return d;
    }

    private static Pair<int[], int[]> sparse_to_csr(Directed A) {

        int[] rp = ArrayUtils.zeros(A.n);
        Pair<int[], int[]> r = A.nonZeroIndex();
        int[] nzj = r.getFirst();
        int[] nzi = r.getSecond();

        for (int n : nzi) {
            rp[n]++;
        }
        cumsum(rp);

        int[] ci = new int[nzi.length];

        for (int i = 0; i < nzi.length; i++) {
            ci[rp[nzi[i] - 1]] = nzj[i];
            rp[nzi[i] - 1]++;
        }

        System.arraycopy(rp, 0, rp, 1, A.n - 1);
        rp[0] = 0;
        for (int i = 0; i < rp.length; i++) {
            rp[i]++;
        }

        return new Pair<int[], int[]>(rp, ci);
    }

    private static void cumsum(int[] rp) {
        for (int i = 1; i < rp.length; i++) {
            rp[i] = rp[i] + rp[i - 1];
        }
    }

    /*
     if ~exist('target','var') || isempty(full), target=0; end

     if isstruct(A), rp=A.rp; ci=A.ci;
     else [rp ci]=sparse_to_csr(A);
     end

     n=length(rp)-1;
     d=-1*ones(n,1); dt=-1*ones(n,1); pred=zeros(1,n);
     sq=zeros(n,1); sqt=0; sqh=0; % search queue and search queue tail/head

     % initCl bfs at u
     sqt=sqt+1; sq(sqt)=u;
     t=0;
     d(u)=0; dt(u)=t; t=t+1; pred(u)=u;
     while sqt-sqh>0
     sqh=sqh+1; v=sq(sqh); % pop v off the head of the queue
     for ri=rp(v):rp(v+1)-1
     w=ci(ri);
     if d(w)<0
     sqt=sqt+1; sq(sqt)=w;
     d(w)=d(v)+1; dt(w)=t; t=t+1; pred(w)=v;
     if w==target, return; end
     end
     end
     end
     */

    public static int findN(int v, int[] phi) {
        for (int i = 0; i < phi.length; i++) {
            if (phi[i] == v) {
                return i;
            }
        }
        return -1;
    }

    private static int[] compute_phi(int[] Q, int n, int k) {

        // compute the relabeling process phi, to transform a tw-tree into a Renyi tw-tree
        // Q must be in increasing order
        int[] phi = ArrayUtils.zeros(n);

        for (int i = 0; i < Q.length; i++) {
            phi[Q[i] - 1] = n - k + (i + 1);
        }

        for (int i = 0; i < n - k; i++) {
            int j = i;

            while (phi[j] != 0) {
                j = phi[j] - 1;
            }

            phi[j] = i + 1;
        }

        return phi;
    }

    public static KTree sample(int n_var, int maxTw, App solver) {
        KTree k = null;

        while (k == null) {
            try {
                k = decode(Dandelion.sample(n_var, maxTw, solver));
            } catch (Exception e) {}
        }
        return k;
    }

    /**
     * Computes the informative score
     *
     * @param M Mutual Information about each variable pair
     * @param S pre-computed scores
     * @return informative scores
     */
    public double informativeScore(MutualInformation M, ParentSet[][] S) {
        double num = mutualInformation(M);

        double den = 0;

        for (int i = 0; i < T.n; i++) {
            // Get the best parent set allowed
            for (ParentSet pSet : S[i]) {
                if (allowedParentSet(i, pSet.parents)) {
                    den += pSet.sk;
                    break;
                }
            }
        }

        return num / Math.abs(den);
    }

    private boolean found(ParentSet pSet, List<int[]> s_p) {
        return index(pSet, s_p) >= 0;
    }

    private int index(ParentSet pSet, List<int[]> s_p) {

        int k = -1;

        for (int j = 0; j < s_p.size(); j++) {
            if (ArrayUtils.sameArray(pSet.parents, s_p.get(j))) {
                k = j;
                break;
            }
        }
        return k;
    }

    /**
     * Searches the allowed parents for thread from the given skeleton
     */

    /*
     public List<int[]> allowedParents(int thread) {

     // Checks the neighbour of v2, if they also are connected to v1
     int[] neigh = T.neighbours(thread);

     List<int[]> allowed = new ArrayList<int[]>();

     // Get all the subsets of the neighbours
     for (int size = 0; size < neigh.length; size++) {

     System.out.println(Arrays.toString(neigh));

     List<int[]> sets = RandomStuff.getSubsets(neigh, size);

     // For each set of the given size
     for (int[] s: sets) {
     Arrays.sort(s);
     boolean complete = true;

     // Check if every pair on the set are connected, we already know that each is already connected to v1 and v2
     for (int i1 = 0; i1 < s.length && complete; i1++) {
     for (int i2 = i1 + 1; i2 < s.length && complete; i2++) {
     if (!T.check(s[i1], s[i2])) {
     complete = false;
     }
     }
     }

     if (complete) {
     allowed.add(s);
     }
     }
     }

     return allowed;
     } */
    public ParentSet[][] selectScores(ParentSet[][] originalPSet) {

        int n = originalPSet.length;

        ParentSet[][] newPSet = new ParentSet[n][];

        for (int i = 0; i < n; i++) {

            List<ParentSet> auxPSet = new ArrayList<ParentSet>();

            // for each parent set
            for (ParentSet pSet : originalPSet[i]) {
                // if it is allowed with this tw-tree add it
                if (allowedParentSet(i, pSet.parents)) {
                    auxPSet.add(pSet);
                }
            }

            // Save the new parent sets list
            newPSet[i] = new ParentSet[auxPSet.size()];
            int j = 0;

            for (ParentSet p : auxPSet) {
                newPSet[i][j++] = p;
            }
        }

        return newPSet;
    }

    private boolean allowedParentSet(int v, int[] ps) {

        // Checks if every parent is a neighbour of thread
        for (int p : ps) {
            if (!T.check(v, p)) {
                return false;
            }
        }

        // Checks if every parent is neighbour of all the others
        for (int i = 0; i < ps.length; i++) {
            for (int j = i + 1; j < ps.length; j++) {
                if (!T.check(ps[i], ps[j])) {
                    return false;
                }
            }
        }

        return true;
    }

    public double score(ParentSet[][] sel) {
        double t = 0;

        for (ParentSet[] s : sel) {
            double zero_sk = s[s.length - 1].sk;

            for (ParentSet p : s) {
                t += p.sk - zero_sk;
            }
        }
        return t;
    }

    private double mutualInformation(MutualInformation M) {

        double t = 0;

        for (int i = 0; i < T.n; i++) {
            for (int j = i + 1; j < T.n; j++) {
                t += (T.check(i, j) ? 1 : 0) * M.getMI(i, j);
            }
        }
        return t;
    }
}
