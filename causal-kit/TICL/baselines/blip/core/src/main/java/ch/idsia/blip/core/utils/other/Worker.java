package ch.idsia.blip.core.utils.other;


public class Worker extends Thread {
    private final Process process;
    public Integer exit;

    public Worker(Process process) {
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
