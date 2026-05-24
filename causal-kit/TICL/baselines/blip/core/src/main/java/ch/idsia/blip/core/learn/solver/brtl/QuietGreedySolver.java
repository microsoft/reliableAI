package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalGreedySearcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalOldSearcher;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;


/**
 * Adds check to out_degree of each node
 */
public class QuietGreedySolver extends BrutalSolver {

    public int max_out_degree;

    @Override
    protected String name() {
        return "Quiet MC sampling";
    }

    @Override
    protected Searcher getSearcher() {
        return new QuietGreedySearcher(this, tw);
    }

    @Override
    public void prepare() {
        super.prepare();
        max_out_degree = tw;
    }

    public class QuietGreedySearcher extends BrutalGreedySearcher {

        int[] out_degree;

        public QuietGreedySearcher(ScoreSolver solver, int tw) {
            super(solver, tw);
        }

        @Override
        protected void clear() {
            super.clear();
            out_degree = new int[n_var];
        }

        @Override
        protected void update(int v, ParentSet ps) {
            super.update(v, ps);

            for (int p : ps.parents) {
                out_degree[p]++;
            }
        }

        @Override
        protected Pair<ParentSet, SIntSet> bestHandler(int v) {

            for (ParentSet p : m_scores[v]) {

                if (!check(p)) {
                    continue;
                }

                for (SIntSet h : handles) {
                    if (containsAll(p.parents, h.set)) {
                        return new Pair<ParentSet, SIntSet>(p, h);
                    }
                }
            }

            return new Pair<ParentSet, SIntSet>(new ParentSet(), new SIntSet());
        }

        protected boolean check(ParentSet p) {

            for (int v : p.parents) {
                if (out_degree[v] > max_out_degree) {
                    // remove_handles(v);
                    return false;
                }
            }

            return true;
        }

    }
}
