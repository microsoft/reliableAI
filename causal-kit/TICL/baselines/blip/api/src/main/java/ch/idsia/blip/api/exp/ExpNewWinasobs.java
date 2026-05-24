package ch.idsia.blip.api.exp;


import ch.idsia.blip.api.learn.solver.win.WinAsobsLearningSolverApi;
import ch.idsia.blip.api.learn.solver.win.WinAsobsSolverApi;
import ch.idsia.blip.core.learn.solver.WinAsobsLearningSolver;

public class ExpNewWinasobs {

    public static void main(String[] args) {
        try {
            new ExpNewWinasobs().test();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void test() {
        String s = "/home/loskana/Desktop/";
        // String n = "net30.d.jkl";
        String n = "pigs.jkl";

        WinAsobsLearningSolverApi.main(
                new String[] {
                        "",
                        "-j", s + n,
                        "-r", s + n.replace(".jkl", ".res"),
                        "-t", "999",
                        "-win", "3",
                        "-b", "1",
                        "-v", "2"
                });

    }
}
