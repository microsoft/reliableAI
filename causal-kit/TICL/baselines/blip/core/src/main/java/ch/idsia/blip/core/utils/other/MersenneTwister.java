package ch.idsia.blip.core.utils.other;


public class MersenneTwister {
    private int mti;
    private int[] mt;
    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = -1727483681;
    private static final int UPPER_MASK = -2147483648;
    private static final int LOWER_MASK = 2147483647;
    private static final int TEMPERING_MASK_B = -1658038656;
    private static final int TEMPERING_MASK_C = -272236544;
    private static final int mag0 = 0;
    private static final int mag1 = -1727483681;
    public static final int DEFAULT_SEED = 4357;

    public MersenneTwister() {
        this(4357);
    }

    public MersenneTwister(int var1) {
        this.mt = new int[624];
        this.setSeed(var1);
    }

    protected void nextBlock() {
        int var1;
        int var2;

        for (var2 = 0; var2 < 227; ++var2) {
            var1 = this.mt[var2] & -2147483648 | this.mt[var2 + 1] & 2147483647;
            this.mt[var2] = this.mt[var2 + 397] ^ var1 >>> 1
                    ^ ((var1 & 1) == 0 ? 0 : -1727483681);
        }

        while (var2 < 623) {
            var1 = this.mt[var2] & -2147483648 | this.mt[var2 + 1] & 2147483647;
            this.mt[var2] = this.mt[var2 + -227] ^ var1 >>> 1
                    ^ ((var1 & 1) == 0 ? 0 : -1727483681);
            ++var2;
        }

        var1 = this.mt[623] & -2147483648 | this.mt[0] & 2147483647;
        this.mt[623] = this.mt[396] ^ var1 >>> 1
                ^ ((var1 & 1) == 0 ? 0 : -1727483681);
        this.mti = 0;
    }

    public int nextInt() {
        if (this.mti == 624) {
            this.nextBlock();
        }

        int var1 = this.mt[this.mti++];

        var1 ^= var1 >>> 11;
        var1 ^= var1 << 7 & -1658038656;
        var1 ^= var1 << 15 & -272236544;
        var1 ^= var1 >>> 18;
        return var1;
    }

    protected void setSeed(int var1) {
        this.mt[0] = var1 & -1;

        for (int var2 = 1; var2 < 624; ++var2) {
            this.mt[var2] = 1812433253
                    * (this.mt[var2 - 1] ^ this.mt[var2 - 1] >> 30)
                            + var2;
            this.mt[var2] &= -1;
        }

        this.mti = 624;
    }

    public double nextDouble() {
        double var1;

        do {
            var1 = ((double) this.nextLong() - -9.223372036854776E18D)
                    * 5.421010862427522E-20D;
        } while (var1 <= 0.0D || var1 >= 1.0D);

        return var1;
    }

    public float nextFloat() {
        float var1;

        do {
            var1 = (float) this.raw();
        } while (var1 >= 1.0F);

        return var1;
    }

    public long nextLong() {
        return ((long) this.nextInt() & 4294967295L) << 32
                | (long) this.nextInt() & 4294967295L;
    }

    public double raw() {
        int var1;

        do {
            var1 = this.nextInt();
        } while (var1 == 0);

        return (double) ((long) var1 & 4294967295L) * 2.3283064365386963E-10D;
    }
}
