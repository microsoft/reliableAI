package ch.idsia.blip.core.learn.scorer;


public class MultiLabelScorer extends IndependenceScorer {

    private final int numClasses;
    private final int numSets;
    private final int numAttributes;

    public MultiLabelScorer(int numAttributes, int numClasses, int numSets) {
        this.numAttributes = numAttributes;
        this.numClasses = numClasses;
        this.numSets = numSets;
    }

    @Override
    public MultiLabelSearcher getNewSearcher(int n) {
        return new MultiLabelSearcher(n);
    }

    public class MultiLabelSearcher extends IndependenceSearcher {

        public MultiLabelSearcher(int in_n) {
            super(in_n);
        }

        @Override
        protected void addParentSetToEvaluate(int[] p1, int p2, int[][] p_values) {

            if (!find(p2, parents)) {
                return;
            }

            // If it's okay, add it normally
            super.addParentSetToEvaluate(p1, p2, p_values);
        }

        /*
         @Override
         protected void selectBestParents() {


         if (n < numClasses * numSets) {
         selectBestClasses();

         } else {

         TIntArrayList prnt = new TIntArrayList();
         // Get parent numClass
         int f = (numAttributes - numClasses * numSets) / numSets;
         int set = (n - numClasses * numSets) / f;
         // pf("%s %s \n ", dat.l_nm_var[n], dat.l_nm_var[set *numClasses]);

         for (int i = 0; i < numClasses; i++)
         prnt.add(set *numClasses + i);

         for (int i = 0; i < f; i++) {
         int v = numClasses * numSets + set * f + i;
         if (v != n)
         prnt.add(v);
         }


         parents = prnt.toArray();

         for (int p : parents) {

         evalSingle(p);
         }
         }

         // pf("%d %s %s\n", n, dat.l_nm_var[n], lst(parents));
         }*/

        private String lst(int[] parents) {
            String s = "[ ";

            for (int p : parents) {
                s += dat.l_nm_var[p] + " ";
            }
            s += "]";
            return s;
        }

        private void selectBestClasses() {
            for (int p = 0; p < numClasses * numSets; p++) {
                if (p != n) {

                    evalSingle(p);
                }
            }

            parents = new int[0];
        }

        private void evalSingle(int p) {
            double sk = score.computeScore(n, p);

            oneScores.put(p, sk);
            addScore(p, sk);
        }

    }
}

