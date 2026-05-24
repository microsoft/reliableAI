package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.learn.solver.WinAsobsImprSolver;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.Iterator;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.RandomStuff.p;


/**
 * Hybrid greedy hill exploration
 */
public class WinAsobsSearcherImprove2 extends WinAsobsSearcher {

    private int a;

    private int b;

    private double worstQueueScore;

    private int max_size;

    private TreeSet<Solution> bests;

    public WinAsobsSearcherImprove2(WinAsobsImprSolver solver) {
        super(solver);
        this.a = solver.a;
        this.b = solver.b;
    }

    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        // First test the new random combination

        if (asobs > 0) {
            asobsOpt();
        }

        go(max_windows);

        // Now try to improve an existing solution

        for (int i = 0; i < b; i++) {

            solver.checkTime();
            if (!solver.still_time) {
                break;
            }

            Solution sol = randSol();

            cloneArray(sol.vars, vars);

            sol.win++;
            p("Trying higher window! " + sol.win);
            go(sol.win);
        }

        return best_str;
    }

    private boolean go(int max_win) {
        initStr();

        // long start = System.currentTimeMillis();

        winasobs();

        // pf("required: %.2f (%d) \n", (System.currentTimeMillis() - start) / 1000.0, max_win);

        return addBestSolution(vars, sk, max_win);
    }

    private Solution randSol() {
        int index = solver.rand.nextInt(bests.size());
        Iterator<Solution> iter = bests.iterator();

        for (int i = 0; i < index; i++) {
            iter.next();
        }
        return iter.next();
    }

    boolean addBestSolution(int[] vars, double sk, int win) {

        boolean toDropWorst = false;

        if (bests.size() > max_size) {

            if (sk < worstQueueScore) {
                // log.conclude("pruned");
                return false;
            }

            toDropWorst = true;
        }

        // Drop worst element in queue, to make room!
        if (toDropWorst) {
            bests.pollLast();
            worstQueueScore = bests.last().sk;
        } else // If we didn't drop any element, check if we have to update the current
        // worst score!
        if (sk < worstQueueScore) {
            worstQueueScore = sk;
        }

        bests.add(new Solution(vars, sk, win));
        return true;
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        bests = new TreeSet<Solution>();

        max_size = a;
    }

    private class Solution implements Comparable<Solution> {

        int[] vars;
        double sk;
        int win = 1;

        public Solution(int[] vars, double sk, int win) {
            this.vars = cloneArray(vars);
            this.sk = sk;
            this.win = win;
        }

        @Override
        public int compareTo(Solution other) {
            if (sk < other.sk) {
                return 1;
            } else {
                return -1;
            }
        }
    }
}
