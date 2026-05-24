package ch.idsia.blip.core.utils.cmd;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class RunTimeout {

    public static int cmdTimeout(final String commandLine,
            final boolean printOutput, final boolean printError, final long timeout) throws IOException {

        Runtime runtime = Runtime.getRuntime();
        Process proc = runtime.exec(commandLine);

        BufferedReader stdInput = new BufferedReader(new
                InputStreamReader(proc.getInputStream()));

        BufferedReader stdError = new BufferedReader(new
                InputStreamReader(proc.getErrorStream()));

        Worker worker = new Worker(proc);

        worker.start();
        try {
            worker.join(timeout);

            StringBuilder out = new StringBuilder();
            String s;

            if (printOutput) {
                while ((s = stdInput.readLine()) != null) {
                    out.append(s);
                }
            }
            if (printError) {
                while ((s = stdError.readLine()) != null) {
                    out.append(s);
                }
            }

            return worker.exit;

        } catch (InterruptedException ex) {
            worker.interrupt();
            Thread.currentThread().interrupt();
        } finally {
            proc.destroy();
        }

        return -1;
    }

    private static class Worker extends Thread {
        private final Process process;
        private Integer exit;

        private Worker(Process process) {
            this.process = process;
        }

        public void run() {
            try {
                exit = process.waitFor();
            } catch (InterruptedException ignore) {
                return;
            }
        }
    }

}
