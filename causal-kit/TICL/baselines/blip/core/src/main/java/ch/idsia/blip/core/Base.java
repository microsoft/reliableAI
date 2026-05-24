package ch.idsia.blip.core;


public class Base {

    private int low_find;

    private int high_find;

    private int mid_find;

    private int midVal_find;

    protected boolean find(int key, int[] a) {
        return pos(key, a) >= 0;
    }

    protected int pos(int key, int[] a) {
        this.low_find = 0;
        this.high_find = (a.length - 1);
        while (this.low_find <= this.high_find) {
            this.mid_find = (this.low_find + this.high_find >>> 1);
            this.midVal_find = a[this.mid_find];
            if (this.midVal_find < key) {
                this.low_find = (this.mid_find + 1);
            } else {
                if (this.midVal_find <= key) {
                    return this.mid_find;
                }
                this.high_find = (this.mid_find - 1);
            }
        }
        return -(this.low_find + 1);
    }
}
