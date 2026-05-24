package ch.idsia.blip.core.utils.math;


import java.io.PrintStream;


/**
 * Faster, more accurate, portable alternative to {@link Math} and
 * {@link StrictMath} for large scale computation.
 * <p>
 * FastMath is a drop-in replacement for both Math and StrictMath. This
 * means that for any method in Math (say {@code Math.sin(x)} or
 * {@code Math.cbrt(y)}), user can directly change the class and use the
 * methods as is (using {@code FastMath.sin(x)} or {@code FastMath.cbrt(y)}
 * in the previous example).
 * </p>
 * <p>
 * FastMath speed is achieved by relying heavily on optimizing compilers
 * to native code present in many JVMs today and use of large tables.
 * The larger tables are lazily initialised on first use, so that the setup
 * time does not penalise methods that don't need them.
 * </p>
 * <p>
 * Note that FastMath is
 * extensively used inside Apache Commons Math, so by calling some algorithms,
 * the overhead when the the tables need to be intialised will occur
 * regardless of the end-user calling FastMath methods directly or not.
 * Performance figures for a specific JVM and hardware can be evaluated by
 * running the FastMathTestPerformance tests in the test directory of the source
 * distribution.
 * </p>
 * <p>
 * FastMath accuracy should be mostly independent of the JVM as it relies only
 * on IEEE-754 basic operations and on embedded tables. Almost all operations
 * are accurate to about 0.5 ulp throughout the domain range. This statement,
 * of course is only a rough global observed behavior, it is <em>not</em> a
 * guarantee for <em>every</em> double numbers input (see William Kahan's <a
 * href="http://en.wikipedia.org/wiki/Rounding#The_table-maker.27s_dilemma">Table
 * Maker's Dilemma</a>).
 * </p>
 *
 * @since 2.2
 */
public class FastMath {

    /**
     * Archimede's constant PI, ratio of circle circumference to diameter.
     */
    public static final double PI = 105414357.0 / 33554432.0
            + 1.984187159361080883e-9;

    /**
     * Napier's constant e, base of the natural logarithm.
     */
    public static final double E = 2850325.0 / 1048576.0
            + 8.254840070411028747e-8;

    /**
     * Index of exp(0) in the array of integer exponentials.
     */
    private static final int EXP_INT_TABLE_MAX_INDEX = 750;

    /**
     * Length of the array of integer exponentials.
     */
    private static final int EXP_INT_TABLE_LEN = EXP_INT_TABLE_MAX_INDEX * 2;

    /**
     * Logarithm table length.
     */
    private static final int LN_MANT_LEN = 1024;

    /**
     * Exponential fractions table length.
     */
    private static final int EXP_FRAC_TABLE_LEN = 1025; // 0, 1/1024, ... 1024/1024

    /**
     * StrictMath.log(Double.MAX_VALUE): {@value}
     */
    private static final double LOG_MAX_VALUE = StrictMath.log(Double.MAX_VALUE);

    /**
     * Indicator for tables initialization.
     * <p>
     * This compile-time constant should be set to true only if one explicitly
     * wants to compute the tables at class loading time instead of using the
     * already computed ones provided as literal arrays below.
     * </p>
     */
    private static final boolean RECOMPUTE_TABLES_AT_RUNTIME = false;

    /**
     * log(2) (high bits).
     */
    private static final double LN_2_A = 0.693147063255310059;

    /**
     * log(2) (low bits).
     */
    private static final double LN_2_B = 1.17304635250823482e-7;

    /**
     * Coefficients for log, when input 0.99 < x < 1.01.
     */
    private static final double LN_QUICK_COEF[][] = {
        { 1.0, 5.669184079525E-24}, { -0.25, -0.25},
        { 0.3333333134651184, 1.986821492305628E-8},
        { -0.25, -6.663542893624021E-14},
        { 0.19999998807907104, 1.1921056801463227E-8},
        { -0.1666666567325592, -7.800414592973399E-9},
        { 0.1428571343421936, 5.650007086920087E-9},
        { -0.12502530217170715, -7.44321345601866E-11},
        { 0.11113807559013367, 9.219544613762692E-9},
    };

    /**
     * Coefficients for log in the range of 1.0 < x < 1.0 + 2^-10.
     */
    private static final double LN_HI_PREC_COEF[][] = {
        { 1.0, -6.032174644509064E-23}, { -0.25, -0.25},
        { 0.3333333134651184, 1.9868161777724352E-8},
        { -0.2499999701976776, -2.957007209750105E-8},
        { 0.19999954104423523, 1.5830993332061267E-10},
        { -0.16624879837036133, -2.6033824355191673E-8}
    };

    /**
     * Sine, Cosine, Tangent tables are for 0, 1/8, 2/8, ... 13/8 = PI/2 approx.
     */
    private static final int SINE_TABLE_LEN = 14;

    /**
     * Sine table (high bits).
     */
    private static final double SINE_TABLE_A[] = {
        +0.0d, +0.1246747374534607d, +0.24740394949913025d, +0.366272509098053d,
        +0.4794255495071411d, +0.5850973129272461d, +0.6816387176513672d,
        +0.7675435543060303d, +0.8414709568023682d, +0.902267575263977d,
        +0.9489846229553223d, +0.9808930158615112d, +0.9974949359893799d,
        +0.9985313415527344d,
    };

    /**
     * Sine table (low bits).
     */
    private static final double SINE_TABLE_B[] = {
        +0.0d, -4.068233003401932E-9d, +9.755392680573412E-9d,
        +1.9987994582857286E-8d, -1.0902938113007961E-8d,
        -3.9986783938944604E-8d, +4.23719669792332E-8d, -5.207000323380292E-8d,
        +2.800552834259E-8d, +1.883511811213715E-8d, -3.5997360512765566E-9d,
        +4.116164446561962E-8d, +5.0614674548127384E-8d, -1.0129027912496858E-9d,
    };

    /**
     * Cosine table (high bits).
     */
    private static final double COSINE_TABLE_A[] = {
        +1.0d, +0.9921976327896118d, +0.9689123630523682d, +0.9305076599121094d,
        +0.8775825500488281d, +0.8109631538391113d, +0.7316888570785522d,
        +0.6409968137741089d, +0.5403022766113281d, +0.4311765432357788d,
        +0.3153223395347595d, +0.19454771280288696d, +0.07073719799518585d,
        -0.05417713522911072d,
    };

    /**
     * Cosine table (low bits).
     */
    private static final double COSINE_TABLE_B[] = {
        +0.0d, +3.4439717236742845E-8d, +5.865827662008209E-8d,
        -3.7999795083850525E-8d, +1.184154459111628E-8d, -3.43338934259355E-8d,
        +1.1795268640216787E-8d, +4.438921624363781E-8d, +2.925681159240093E-8d,
        -2.6437112632041807E-8d, +2.2860509143963117E-8d, -4.813899778443457E-9d,
        +3.6725170580355583E-9d, +2.0217439756338078E-10d,
    };

    /**
     * Tangent table, used by atan() (high bits).
     */
    private static final double TANGENT_TABLE_A[] = {
        +0.0d, +0.1256551444530487d, +0.25534194707870483d, +0.3936265707015991d,
        +0.5463024377822876d, +0.7214844226837158d, +0.9315965175628662d,
        +1.1974215507507324d, +1.5574076175689697d, +2.092571258544922d,
        +3.0095696449279785d, +5.041914939880371d, +14.101419448852539d,
        -18.430862426757812d,
    };

    /**
     * Tangent table, used by atan() (low bits).
     */
    private static final double TANGENT_TABLE_B[] = {
        +0.0d, -7.877917738262007E-9d, -2.5857668567479893E-8d,
        +5.2240336371356666E-9d, +5.206150291559893E-8d, +1.8307188599677033E-8d,
        -5.7618793749770706E-8d, +7.848361555046424E-8d, +1.0708593250394448E-7d,
        +1.7827257129423813E-8d, +2.893485277253286E-8d, +3.1660099222737955E-7d,
        +4.983191803254889E-7d, -3.356118100840571E-7d,
    };

    /**
     * 0x40000000 - used to split a double into two parts, both with the low order bits cleared.
     * Equivalent to 2^30.
     */
    private static final long HEX_40000000 = 0x40000000L; // 1073741824L

    /**
     * Mask used to clear the non-sign part of an int.
     */
    private static final int MASK_NON_SIGN_INT = 0x7fffffff;

    /**
     * Mask used to clear the non-sign part of a long.
     */
    private static final long MASK_NON_SIGN_LONG = 0x7fffffffffffffffL;

    /**
     * Mask used to extract exponent from double bits.
     */
    private static final long MASK_DOUBLE_EXPONENT = 0x7ff0000000000000L;

    /**
     * Mask used to extract mantissa from double bits.
     */
    private static final long MASK_DOUBLE_MANTISSA = 0x000fffffffffffffL;

    /**
     * Mask used to add implicit high order bit for normalized double.
     */
    private static final long IMPLICIT_HIGH_BIT = 0x0010000000000000L;

    /**
     * 2^52 - double numbers this large must be integral (no fraction) or NaN or Infinite
     */
    private static final double TWO_POWER_52 = 4503599627370496.0;

    /**
     * Private Constructor
     */
    private FastMath() {}

    // Generic helper methods


    /**
     * Compute the square root of a number.
     * <p><b>Note:</b> this implementation currently delegates to {@link Math#sqrt}
     *
     * @param a number on which evaluation is done
     * @return square root of a
     */
    public static double sqrt(final double a) {
        return Math.sqrt(a);
    }

    /**
     * Returns a pseudo-random number between 0.0 and 1.0.
     * <p><b>Note:</b> this implementation currently delegates to {@link Math#random}
     *
     * @return a random number between 0.0 and 1.0
     */
    public static double random() {
        return Math.random();
    }

    /**
     * Internal helper method for exponential function.
     *
     * @param x      original argument of the exponential function
     * @param extra  extra bits of precision on input (To Be Confirmed)
     * @param hiPrec extra bits of precision on output (To Be Confirmed)
     * @return exp(x)
     */
    private static double exp(double x, double extra, double[] hiPrec) {
        double intPartA;
        double intPartB;
        int intVal = (int) x;

        /* Lookup exp(floor(x)).
         * intPartA will have the upper 22 bits, intPartB will have the lower
         * 52 bits.
         */
        if (x < 0.0) {

            // We don't check against intVal here as conversion of large negative double values
            // may be affected by a JIT bug. Subsequent comparisons can safely use intVal
            if (x < -746d) {
                if (hiPrec != null) {
                    hiPrec[0] = 0.0;
                    hiPrec[1] = 0.0;
                }
                return 0.0;
            }

            if (intVal < -709) {

                /* This will produce a subnormal output */
                final double result = exp(x + 40.19140625, extra, hiPrec)
                        / 285040095144011776.0;

                if (hiPrec != null) {
                    hiPrec[0] /= 285040095144011776.0;
                    hiPrec[1] /= 285040095144011776.0;
                }
                return result;
            }

            if (intVal == -709) {

                /* exp(1.494140625) is nearly a machine number... */
                final double result = exp(x + 1.494140625, extra, hiPrec)
                        / 4.455505956692756620;

                if (hiPrec != null) {
                    hiPrec[0] /= 4.455505956692756620;
                    hiPrec[1] /= 4.455505956692756620;
                }
                return result;
            }

            intVal--;

        } else {
            if (intVal > 709) {
                if (hiPrec != null) {
                    hiPrec[0] = Double.POSITIVE_INFINITY;
                    hiPrec[1] = 0.0;
                }
                return Double.POSITIVE_INFINITY;
            }

        }

        intPartA = ExpIntTable.EXP_INT_TABLE_A[EXP_INT_TABLE_MAX_INDEX + intVal];
        intPartB = ExpIntTable.EXP_INT_TABLE_B[EXP_INT_TABLE_MAX_INDEX + intVal];

        /* Get the fractional part of x, find the greatest multiple of 2^-10 less than
         * x and look up the exp function of it.
         * fracPartA will have the upper 22 bits, fracPartB the lower 52 bits.
         */
        final int intFrac = (int) ((x - intVal) * 1024.0);
        final double fracPartA = ExpFracTable.EXP_FRAC_TABLE_A[intFrac];
        final double fracPartB = ExpFracTable.EXP_FRAC_TABLE_B[intFrac];

        /* epsilon is the difference in x from the nearest multiple of 2^-10.  It
         * has a value in the range 0 <= epsilon < 2^-10.
         * Do the subtraction from x as the last step to avoid possible loss of precision.
         */
        final double epsilon = x - (intVal + intFrac / 1024.0);

        /* Compute z = exp(epsilon) - 1.0 via a minimax polynomial.  z has
         full double precision (52 bits).  Since z < 2^-10, we will have
         62 bits of precision when combined with the constant 1.  This will be
         used in the last addition below to get proper rounding. */

        /* Remez generated polynomial.  Converges on the interval [0, 2^-10], error
         is less than 0.5 ULP */
        double z = 0.04168701738764507;

        z = z * epsilon + 0.1666666505023083;
        z = z * epsilon + 0.5000000000042687;
        z = z * epsilon + 1.0;
        z = z * epsilon + -3.940510424527919E-20;

        /* Compute (intPartA+intPartB) * (fracPartA+fracPartB) by binomial
         expansion.
         tempA is exact since intPartA and intPartB only have 22 bits each.
         tempB will have 52 bits of precision.
         */
        double tempA = intPartA * fracPartA;
        double tempB = intPartA * fracPartB + intPartB * fracPartA
                + intPartB * fracPartB;

        /* Compute the result.  (1+z)(tempA+tempB).  Order of operations is
         important.  For accuracy add by increasing size.  tempA is exact and
         much larger than the others.  If there are extra bits specified from the
         pow() function, use them. */
        final double tempC = tempB + tempA;

        // If tempC is positive infinite, the evaluation below could result in NaN,
        // because z could be negative at the same time.
        if (tempC == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }

        final double result;

        if (extra != 0.0) {
            result = tempC * extra * z + tempC * extra + tempC * z + tempB
                    + tempA;
        } else {
            result = tempC * z + tempB + tempA;
        }

        if (hiPrec != null) {
            // If requesting high precision
            hiPrec[0] = tempA;
            hiPrec[1] = tempC * extra * z + tempC * extra + tempC * z + tempB;
        }

        return result;
    }

    /**
     * Natural logarithm.
     *
     * @param x a double
     * @return log(x)
     */
    public static double log(final double x) {
        return log(x, null);
    }

    /**
     * Internal helper method for natural logarithm function.
     *
     * @param x      original argument of the natural logarithm function
     * @param hiPrec extra bits of precision on output (To Be Confirmed)
     * @return log(x)
     */
    private static double log(final double x, final double[] hiPrec) {
        if (x == 0) { // Handle special case of +0/-0
            return Double.NEGATIVE_INFINITY;
        }
        long bits = Double.doubleToRawLongBits(x);

        /* Handle special cases of negative input, and NaN */
        if (((bits & 0x8000000000000000L) != 0 || x != x) && x != 0.0) {
            if (hiPrec != null) {
                hiPrec[0] = Double.NaN;
            }

            return Double.NaN;
        }

        /* Handle special cases of Positive infinity. */
        if (x == Double.POSITIVE_INFINITY) {
            if (hiPrec != null) {
                hiPrec[0] = Double.POSITIVE_INFINITY;
            }

            return Double.POSITIVE_INFINITY;
        }

        /* Extract the exponent */
        int exp = (int) (bits >> 52) - 1023;

        if ((bits & 0x7ff0000000000000L) == 0) {
            // Subnormal!
            if (x == 0) {
                // Zero
                if (hiPrec != null) {
                    hiPrec[0] = Double.NEGATIVE_INFINITY;
                }

                return Double.NEGATIVE_INFINITY;
            }

            /* Normalize the subnormal number. */
            bits <<= 1;
            while ((bits & 0x0010000000000000L) == 0) {
                --exp;
                bits <<= 1;
            }
        }

        if ((exp == -1 || exp == 0) && x < 1.01 && x > 0.99 && hiPrec == null) {

            /* The normal method doesn't work well in the range [0.99, 1.01], so call do a straight
             polynomial expansion in higer precision. */

            /* Compute x - 1.0 and split it */
            double xa = x - 1.0;
            double xb;
            double tmp = xa * HEX_40000000;
            double aa = xa + tmp - tmp;
            double ab = xa - aa;

            xa = aa;
            xb = ab;

            final double[] lnCoef_last = LN_QUICK_COEF[LN_QUICK_COEF.length - 1];
            double ya = lnCoef_last[0];
            double yb = lnCoef_last[1];

            for (int i = LN_QUICK_COEF.length - 2; i >= 0; i--) {

                /* Multiply a = y * x */
                aa = ya * xa;
                ab = ya * xb + yb * xa + yb * xb;

                /* split, so now y = a */
                tmp = aa * HEX_40000000;
                ya = aa + tmp - tmp;
                yb = aa - ya + ab;

                /* Add  a = y + lnQuickCoef */
                final double[] lnCoef_i = LN_QUICK_COEF[i];

                aa = ya + lnCoef_i[0];
                ab = yb + lnCoef_i[1];

                /* Split y = a */
                tmp = aa * HEX_40000000;
                ya = aa + tmp - tmp;
                yb = aa - ya + ab;
            }

            /* Multiply a = y * x */
            aa = ya * xa;
            ab = ya * xb + yb * xa + yb * xb;

            /* split, so now y = a */
            tmp = aa * HEX_40000000;
            ya = aa + tmp - tmp;
            yb = aa - ya + ab;

            return ya + yb;
        }

        // lnm is a log of a number in the range of 1.0 - 2.0, so 0 <= lnm < ln(2)
        final double[] lnm = lnMant.LN_MANT[(int) ((bits & 0x000ffc0000000000L) >> 42)];

        /*
         double epsilon = x / Double.longBitsToDouble(bits & 0xfffffc0000000000L);

         epsilon -= 1.0;
         */

        // y is the most significant 10 bits of the mantissa
        // double y = Double.longBitsToDouble(bits & 0xfffffc0000000000L);
        // double epsilon = (x - y) / y;
        final double epsilon = (bits & 0x3ffffffffffL)
                / (TWO_POWER_52 + (bits & 0x000ffc0000000000L));

        double lnza;
        double lnzb = 0.0;

        if (hiPrec != null) {

            /* split epsilon -> x */
            double tmp = epsilon * HEX_40000000;
            double aa = epsilon + tmp - tmp;
            double ab = epsilon - aa;
            double xa = aa;
            double xb = ab;

            /* Need a more accurate epsilon, so adjust the division. */
            final double numer = bits & 0x3ffffffffffL;
            final double denom = TWO_POWER_52 + (bits & 0x000ffc0000000000L);

            aa = numer - xa * denom - xb * denom;
            xb += aa / denom;

            /* Remez polynomial evaluation */
            final double[] lnCoef_last = LN_HI_PREC_COEF[LN_HI_PREC_COEF.length - 1];
            double ya = lnCoef_last[0];
            double yb = lnCoef_last[1];

            for (int i = LN_HI_PREC_COEF.length - 2; i >= 0; i--) {

                /* Multiply a = y * x */
                aa = ya * xa;
                ab = ya * xb + yb * xa + yb * xb;

                /* split, so now y = a */
                tmp = aa * HEX_40000000;
                ya = aa + tmp - tmp;
                yb = aa - ya + ab;

                /* Add  a = y + lnHiPrecCoef */
                final double[] lnCoef_i = LN_HI_PREC_COEF[i];

                aa = ya + lnCoef_i[0];
                ab = yb + lnCoef_i[1];

                /* Split y = a */
                tmp = aa * HEX_40000000;
                ya = aa + tmp - tmp;
                yb = aa - ya + ab;
            }

            /* Multiply a = y * x */
            aa = ya * xa;
            ab = ya * xb + yb * xa + yb * xb;

            /* split, so now lnz = a */
            
            /*
             tmp = aa * 1073741824.0;
             lnza = aa + tmp - tmp;
             lnzb = aa - lnza + ab;
             */
            lnza = aa + ab;
            lnzb = -(lnza - aa - ab);
        } else {

            /* High precision not required.  Eval Remez polynomial
             using standard double precision */
            lnza = -0.16624882440418567;
            lnza = lnza * epsilon + 0.19999954120254515;
            lnza = lnza * epsilon + -0.2499999997677497;
            lnza = lnza * epsilon + 0.3333333333332802;
            lnza = lnza * epsilon + -0.5;
            lnza = lnza * epsilon + 1.0;
            lnza *= epsilon;
        }

        /* Relative sizes:
         * lnzb     [0, 2.33E-10]
         * lnm[1]   [0, 1.17E-7]
         * ln2B*exp [0, 1.12E-4]
         * lnza      [0, 9.7E-4]
         * lnm[0]   [0, 0.692]
         * ln2A*exp [0, 709]
         */

        /* Compute the following sum:
         * lnzb + lnm[1] + ln2B*exp + lnza + lnm[0] + ln2A*exp;
         */

        // return lnzb + lnm[1] + ln2B*exp + lnza + lnm[0] + ln2A*exp;
        double a = LN_2_A * exp;
        double b = 0.0;
        double c = a + lnm[0];
        double d = -(c - a - lnm[0]);

        a = c;
        b += d;

        c = a + lnza;
        d = -(c - a - lnza);
        a = c;
        b += d;

        c = a + LN_2_B * exp;
        d = -(c - a - LN_2_B * exp);
        a = c;
        b += d;

        c = a + lnm[1];
        d = -(c - a - lnm[1]);
        a = c;
        b += d;

        c = a + lnzb;
        d = -(c - a - lnzb);
        a = c;
        b += d;

        if (hiPrec != null) {
            hiPrec[0] = a;
            hiPrec[1] = b;
        }

        return a + b;
    }

    /**
     * Computes the <a href="http://mathworld.wolfram.com/Logarithm.html">
     * logarithm</a> in a given base.
     * <p>
     * Returns {@code NaN} if either argument is negative.
     * If {@code base} is 0 and {@code x} is positive, 0 is returned.
     * If {@code base} is positive and {@code x} is 0,
     * {@code Double.NEGATIVE_INFINITY} is returned.
     * If both arguments are 0, the result is {@code NaN}.
     *
     * @param base App of the logarithm, must be greater than 0.
     * @param x    Argument, must be greater than 0.
     * @return the value of the logarithm, thread.e. the number {@code y} such that
     * <code>base<sup>y</sup> = x</code>.
     * @since 1.2 (previously in {@code MathUtils}, moved as of version 3.0)
     */
    public static double log(double base, double x) {
        return log(x) / log(base);
    }

    /**
     * Power function.  Compute x^y.
     *
     * @param x a double
     * @param y a double
     * @return double
     */
    public static double pow(final double x, final double y) {

        if (y == 0) {
            // y = -0 or y = +0
            return 1.0;
        } else {

            final long yBits = Double.doubleToRawLongBits(y);
            final int yRawExp = (int) ((yBits & MASK_DOUBLE_EXPONENT) >> 52);
            final long yRawMantissa = yBits & MASK_DOUBLE_MANTISSA;
            final long xBits = Double.doubleToRawLongBits(x);
            final int xRawExp = (int) ((xBits & MASK_DOUBLE_EXPONENT) >> 52);
            final long xRawMantissa = xBits & MASK_DOUBLE_MANTISSA;

            if (yRawExp > 1085) {
                // y is either a very large integral value that does not fit in a long or it is a special number

                if ((yRawExp == 2047 && yRawMantissa != 0)
                        || (xRawExp == 2047 && xRawMantissa != 0)) {
                    // NaN
                    return Double.NaN;
                } else if (xRawExp == 1023 && xRawMantissa == 0) {
                    // x = -1.0 or x = +1.0
                    if (yRawExp == 2047) {
                        // y is infinite
                        return Double.NaN;
                    } else {
                        // y is a large even integer
                        return 1.0;
                    }
                } else {
                    // the absolute value of x is either greater or smaller than 1.0

                    // if yRawExp == 2047 and mantissa is 0, y = -infinity or y = +infinity
                    // if 1085 < yRawExp < 2047, y is simply a large number, however, due to limited
                    // accuracy, at this magnitude it behaves just like infinity with regards to x
                    if ((y > 0) ^ (xRawExp < 1023)) {
                        // either y = +infinity (or large engouh) and abs(x) > 1.0
                        // or     y = -infinity (or large engouh) and abs(x) < 1.0
                        return Double.POSITIVE_INFINITY;
                    } else {
                        // either y = +infinity (or large engouh) and abs(x) < 1.0
                        // or     y = -infinity (or large engouh) and abs(x) > 1.0
                        return +0.0;
                    }
                }

            } else {
                // y is a regular non-zero number

                if (yRawExp >= 1023) {
                    // y may be an integral value, which should be handled specifically
                    final long yFullMantissa = IMPLICIT_HIGH_BIT | yRawMantissa;

                    if (yRawExp < 1075) {
                        // normal number with negative shift that may have a fractional part
                        final long integralMask = (-1L) << (1075 - yRawExp);

                        if ((yFullMantissa & integralMask) == yFullMantissa) {
                            // all fractional bits are 0, the number is really integral
                            final long l = yFullMantissa >> (1075 - yRawExp);

                            return FastMath.pow(x, (y < 0) ? -l : l);
                        }
                    } else {
                        // normal number with positive shift, always an integral value
                        // we know it fits in a primitive long because yRawExp > 1085 has been handled above
                        final long l = yFullMantissa << (yRawExp - 1075);

                        return FastMath.pow(x, (y < 0) ? -l : l);
                    }
                }

                // y is a non-integral value

                if (x == 0) {
                    // x = -0 or x = +0
                    // the integer powers have already been handled above
                    return y < 0 ? Double.POSITIVE_INFINITY : +0.0;
                } else if (xRawExp == 2047) {
                    if (xRawMantissa == 0) {
                        // x = -infinity or x = +infinity
                        return (y < 0) ? +0.0 : Double.POSITIVE_INFINITY;
                    } else {
                        // NaN
                        return Double.NaN;
                    }
                } else if (x < 0) {
                    // the integer powers have already been handled above
                    return Double.NaN;
                } else {

                    // this is the general case, for regular fractional numbers x and y

                    // Split y into ya and yb such that y = ya+yb
                    final double tmp = y * HEX_40000000;
                    final double ya = (y + tmp) - tmp;
                    final double yb = y - ya;

                    /* Compute ln(x) */
                    final double lns[] = new double[2];
                    final double lores = log(x, lns);

                    if (Double.isInfinite(lores)) { // don't allow this to be converted to NaN
                        return lores;
                    }

                    double lna = lns[0];
                    double lnb = lns[1];

                    /* resplit lns */
                    final double tmp1 = lna * HEX_40000000;
                    final double tmp2 = (lna + tmp1) - tmp1;

                    lnb += lna - tmp2;
                    lna = tmp2;

                    // y*ln(x) = (aa+ab)
                    final double aa = lna * ya;
                    final double ab = lna * yb + lnb * ya + lnb * yb;

                    lna = aa + ab;
                    lnb = -(lna - aa - ab);

                    double z = 1.0 / 120.0;

                    z = z * lnb + (1.0 / 24.0);
                    z = z * lnb + (1.0 / 6.0);
                    z = z * lnb + 0.5;
                    z = z * lnb + 1.0;
                    z *= lnb;

                    final double result = exp(lna, z, null);

                    // result = result + result * z;
                    return result;

                }
            }

        }

    }

    /**
     * Raise a double to an int power.
     *
     * @param d Number to raise.
     * @param e Exponent.
     * @return d<sup>e</sup>
     * @since 3.1
     */
    public static double pow(double d, int e) {
        return pow(d, (long) e);
    }

    /**
     * Raise a double to a long power.
     *
     * @param d Number to raise.
     * @param e Exponent.
     * @return d<sup>e</sup>
     * @since 3.6
     */
    private static double pow(double d, long e) {
        if (e == 0) {
            return 1.0;
        } else if (e > 0) {
            return new Split(d).pow(e).full;
        } else {
            return new Split(d).reciprocal().pow(-e).full;
        }
    }

    /**
     * Class operator on double numbers split into one 26 bits number and one 27 bits number.
     */
    private static class Split {

        /**
         * Split version of NaN.
         */
        public static final Split NAN = new Split(Double.NaN, 0);

        /**
         * Split version of positive infinity.
         */
        public static final Split POSITIVE_INFINITY = new Split(
                Double.POSITIVE_INFINITY, 0);

        /**
         * Split version of negative infinity.
         */
        public static final Split NEGATIVE_INFINITY = new Split(
                Double.NEGATIVE_INFINITY, 0);

        /**
         * Full number.
         */
        private final double full;

        /**
         * High order bits.
         */
        private final double high;

        /**
         * Low order bits.
         */
        private final double low;

        /**
         * Simple constructor.
         *
         * @param x number to split
         */
        Split(final double x) {
            full = x;
            high = Double.longBitsToDouble(
                    Double.doubleToRawLongBits(x) & ((-1L) << 27));
            low = x - high;
        }

        /**
         * Simple constructor.
         *
         * @param high high order bits
         * @param low  low order bits
         */
        Split(final double high, final double low) {
            this(
                    high == 0.0
                            ? (low == 0.0
                                    && Double.doubleToRawLongBits(high)
                                            == Long.MIN_VALUE /* negative zero */
                                                    ? -0.0
                                                    : low)
                                                    : high + low,
                                                    high,
                                                    low);
        }

        /**
         * Simple constructor.
         *
         * @param full full number
         * @param high high order bits
         * @param low  low order bits
         */
        Split(final double full, final double high, final double low) {
            this.full = full;
            this.high = high;
            this.low = low;
        }

        /**
         * Multiply the instance by another one.
         *
         * @param b other instance to multiply by
         * @return product
         */
        public Split multiply(final Split b) {
            // beware the following expressions must NOT be simplified, they rely on floating point arithmetic properties
            final Split mulBasic = new Split(full * b.full);
            final double mulError = low * b.low
                    - (((mulBasic.full - high * b.high) - low * b.high)
                            - high * b.low);

            return new Split(mulBasic.high, mulBasic.low + mulError);
        }

        /**
         * Compute the reciprocal of the instance.
         *
         * @return reciprocal of the instance
         */
        public Split reciprocal() {

            final double approximateInv = 1.0 / full;
            final Split splitInv = new Split(approximateInv);

            // if 1.0/d were computed perfectly, remultiplying it by d should give 1.0
            // we want to PR the error so we can fix the low order bits of approximateInvLow
            // beware the following expressions must NOT be simplified, they rely on floating point arithmetic properties
            final Split product = multiply(splitInv);
            final double error = (product.high - 1) + product.low;

            // better accuracy PR of reciprocal
            return Double.isNaN(error)
                    ? splitInv
                    : new Split(splitInv.high, splitInv.low - error / full);

        }

        /**
         * Computes this^e.
         *
         * @param e exponent (beware, here it MUST be > 0; the only exclusion is Long.MIN_VALUE)
         * @return d^e, split in high and low bits
         * @since 3.6
         */
        private Split pow(final long e) {

            // init result
            Split result = new Split(1);

            // d^(2p)
            Split d2p = new Split(full, high, low);

            for (long p = e; p != 0; p >>>= 1) {

                if ((p & 0x1) != 0) {
                    // accurate multiplication result = result * d^(2p) using Veltkamp TwoProduct algorithm
                    result = result.multiply(d2p);
                }

                // accurate squaring d^(2(p+1)) = d^(2p) * d^(2p) using Veltkamp TwoProduct algorithm
                d2p = d2p.multiply(d2p);

            }

            if (Double.isNaN(result.full)) {
                if (Double.isNaN(full)) {
                    return Split.NAN;
                } else {
                    // some intermediate numbers exceeded capacity,
                    // and the low order bits became NaN (because infinity - infinity = NaN)
                    if (FastMath.abs(full) < 1) {
                        return new Split(FastMath.copySign(0.0, full), 0.0);
                    } else if (full < 0 && (e & 0x1) == 1) {
                        return Split.NEGATIVE_INFINITY;
                    } else {
                        return Split.POSITIVE_INFINITY;
                    }
                }
            } else {
                return result;
            }

        }

    }

    /**
     * Absolute value.
     *
     * @param x number from which absolute value is requested
     * @return abs(x)
     */
    public static int abs(final int x) {
        final int i = x >>> 31;

        return (x ^ (~i + 1)) + i;
    }

    /**
     * Absolute value.
     *
     * @param x number from which absolute value is requested
     * @return abs(x)
     */
    public static long abs(final long x) {
        final long l = x >>> 63;

        // l is one if x negative zero else
        // ~l+1 is zero if x is positive, -1 if x is negative
        // x^(~l+1) is x is x is positive, ~x if x is negative
        // add around
        return (x ^ (~l + 1)) + l;
    }

    /**
     * Absolute value.
     *
     * @param x number from which absolute value is requested
     * @return abs(x)
     */
    public static float abs(final float x) {
        return Float.intBitsToFloat(
                MASK_NON_SIGN_INT & Float.floatToRawIntBits(x));
    }

    /**
     * Absolute value.
     *
     * @param x number from which absolute value is requested
     * @return abs(x)
     */
    public static double abs(double x) {
        return Double.longBitsToDouble(
                MASK_NON_SIGN_LONG & Double.doubleToRawLongBits(x));
    }

    /**
     * Compute the minimum of two values
     *
     * @param a first value
     * @param b second value
     * @return a if a is lesser or equal to b, b otherwise
     */
    public static int min(final int a, final int b) {
        return (a <= b) ? a : b;
    }

    /**
     * Compute the minimum of two values
     *
     * @param a first value
     * @param b second value
     * @return a if a is lesser or equal to b, b otherwise
     */
    public static long min(final long a, final long b) {
        return (a <= b) ? a : b;
    }

    /**
     * Compute the minimum of two values
     *
     * @param a first value
     * @param b second value
     * @return a if a is lesser or equal to b, b otherwise
     */
    public static float min(final float a, final float b) {
        if (a > b) {
            return b;
        }
        if (a < b) {
            return a;
        }

        /* if either arg is NaN, return NaN */
        if (a != b) {
            return Float.NaN;
        }

        /* min(+0.0,-0.0) == -0.0 */
        
        /* 0x80000000 == Float.floatToRawIntBits(-0.0d) */
        int bits = Float.floatToRawIntBits(a);

        if (bits == 0x80000000) {
            return a;
        }
        return b;
    }

    /**
     * Compute the minimum of two values
     *
     * @param a first value
     * @param b second value
     * @return a if a is lesser or equal to b, b otherwise
     */
    public static double min(final double a, final double b) {
        if (a > b) {
            return b;
        }
        if (a < b) {
            return a;
        }

        /* if either arg is NaN, return NaN */
        if (a != b) {
            return Double.NaN;
        }

        /* min(+0.0,-0.0) == -0.0 */
        
        /* 0x8000000000000000L == Double.doubleToRawLongBits(-0.0d) */
        long bits = Double.doubleToRawLongBits(a);

        if (bits == 0x8000000000000000L) {
            return a;
        }
        return b;
    }

    /**
     * Compute the maximum of two values
     *
     * @param a first value
     * @param b second value
     * @return b if a is lesser or equal to b, a otherwise
     */
    public static int max(final int a, final int b) {
        return (a <= b) ? b : a;
    }

    /**
     * Compute the maximum of two values
     *
     * @param a first value
     * @param b second value
     * @return b if a is lesser or equal to b, a otherwise
     */
    public static long max(final long a, final long b) {
        return (a <= b) ? b : a;
    }

    /**
     * Compute the maximum of two values
     *
     * @param a first value
     * @param b second value
     * @return b if a is lesser or equal to b, a otherwise
     */
    public static float max(final float a, final float b) {
        if (a > b) {
            return a;
        }
        if (a < b) {
            return b;
        }

        /* if either arg is NaN, return NaN */
        if (a != b) {
            return Float.NaN;
        }

        /* min(+0.0,-0.0) == -0.0 */
        
        /* 0x80000000 == Float.floatToRawIntBits(-0.0d) */
        int bits = Float.floatToRawIntBits(a);

        if (bits == 0x80000000) {
            return b;
        }
        return a;
    }

    /**
     * Returns the first argument with the sign of the second argument.
     * A NaN {@code sign} argument is treated as positive.
     *
     * @param magnitude the value to return
     * @param sign      the sign for the returned value
     * @return the magnitude with the same sign as the {@code sign} argument
     */
    private static double copySign(double magnitude, double sign) {
        // The highest order bit is going to be zero if the
        // highest order bit of m and s is the same and one otherwise.
        // So (m^s) will be positive if both m and s have the same sign
        // and negative otherwise.
        final long m = Double.doubleToRawLongBits(magnitude); // don't care about NaN
        final long s = Double.doubleToRawLongBits(sign);

        if ((m ^ s) >= 0) {
            return magnitude;
        }
        return -magnitude; // flip sign
    }

    /**
     * Print out contents of arrays, and check the length.
     * <p>used to generate the preset arrays originally.</p>
     *
     * @param a unused
     */
    public static void main(String[] a) throws Exception {
        PrintStream out = System.out;

        FastMathCalc.printarray(out, "EXP_INT_TABLE_A", EXP_INT_TABLE_LEN,
                ExpIntTable.EXP_INT_TABLE_A);
        FastMathCalc.printarray(out, "EXP_INT_TABLE_B", EXP_INT_TABLE_LEN,
                ExpIntTable.EXP_INT_TABLE_B);
        FastMathCalc.printarray(out, "EXP_FRAC_TABLE_A", EXP_FRAC_TABLE_LEN,
                ExpFracTable.EXP_FRAC_TABLE_A);
        FastMathCalc.printarray(out, "EXP_FRAC_TABLE_B", EXP_FRAC_TABLE_LEN,
                ExpFracTable.EXP_FRAC_TABLE_B);
        FastMathCalc.printarray(out, "LN_MANT", LN_MANT_LEN, lnMant.LN_MANT);
        FastMathCalc.printarray(out, "SINE_TABLE_A", SINE_TABLE_LEN,
                SINE_TABLE_A);
        FastMathCalc.printarray(out, "SINE_TABLE_B", SINE_TABLE_LEN,
                SINE_TABLE_B);
        FastMathCalc.printarray(out, "COSINE_TABLE_A", SINE_TABLE_LEN,
                COSINE_TABLE_A);
        FastMathCalc.printarray(out, "COSINE_TABLE_B", SINE_TABLE_LEN,
                COSINE_TABLE_B);
        FastMathCalc.printarray(out, "TANGENT_TABLE_A", SINE_TABLE_LEN,
                TANGENT_TABLE_A);
        FastMathCalc.printarray(out, "TANGENT_TABLE_B", SINE_TABLE_LEN,
                TANGENT_TABLE_B);
    }

    /**
     * Enclose large data table in nested static class so it's only loaded on first access.
     */
    private static class ExpIntTable {

        /**
         * Exponential evaluated at integer values,
         * exp(x) =  expIntTableA[x + EXP_INT_TABLE_MAX_INDEX] + expIntTableB[x+EXP_INT_TABLE_MAX_INDEX].
         */
        private static final double[] EXP_INT_TABLE_A;

        /**
         * Exponential evaluated at integer values,
         * exp(x) =  expIntTableA[x + EXP_INT_TABLE_MAX_INDEX] + expIntTableB[x+EXP_INT_TABLE_MAX_INDEX]
         */
        private static final double[] EXP_INT_TABLE_B;

        static {
            if (RECOMPUTE_TABLES_AT_RUNTIME) {
                EXP_INT_TABLE_A = new double[FastMath.EXP_INT_TABLE_LEN];
                EXP_INT_TABLE_B = new double[FastMath.EXP_INT_TABLE_LEN];

                final double tmp[] = new double[2];
                final double recip[] = new double[2];

                // Populate expIntTable
                for (int i = 0; i < FastMath.EXP_INT_TABLE_MAX_INDEX; i++) {
                    FastMathCalc.expint(i, tmp);
                    EXP_INT_TABLE_A[i + FastMath.EXP_INT_TABLE_MAX_INDEX] = tmp[0];
                    EXP_INT_TABLE_B[i + FastMath.EXP_INT_TABLE_MAX_INDEX] = tmp[1];

                    if (i != 0) {
                        // Negative integer powers
                        FastMathCalc.splitReciprocal(tmp, recip);
                        EXP_INT_TABLE_A[FastMath.EXP_INT_TABLE_MAX_INDEX - i] = recip[0];
                        EXP_INT_TABLE_B[FastMath.EXP_INT_TABLE_MAX_INDEX - i] = recip[1];
                    }
                }
            } else {
                EXP_INT_TABLE_A = FastMathLiteralArrays.loadExpIntA();
                EXP_INT_TABLE_B = FastMathLiteralArrays.loadExpIntB();
            }
        }
    }


    /**
     * Enclose large data table in nested static class so it's only loaded on first access.
     */
    private static class ExpFracTable {

        /**
         * Exponential over the range of 0 - 1 in increments of 2^-10
         * exp(x/1024) =  expFracTableA[x] + expFracTableB[x].
         * 1024 = 2^10
         */
        private static final double[] EXP_FRAC_TABLE_A;

        /**
         * Exponential over the range of 0 - 1 in increments of 2^-10
         * exp(x/1024) =  expFracTableA[x] + expFracTableB[x].
         */
        private static final double[] EXP_FRAC_TABLE_B;

        static {
            if (RECOMPUTE_TABLES_AT_RUNTIME) {
                EXP_FRAC_TABLE_A = new double[FastMath.EXP_FRAC_TABLE_LEN];
                EXP_FRAC_TABLE_B = new double[FastMath.EXP_FRAC_TABLE_LEN];

                final double tmp[] = new double[2];

                // Populate expFracTable
                final double factor = 1d / (EXP_FRAC_TABLE_LEN - 1);

                for (int i = 0; i < EXP_FRAC_TABLE_A.length; i++) {
                    FastMathCalc.slowexp(i * factor, tmp);
                    EXP_FRAC_TABLE_A[i] = tmp[0];
                    EXP_FRAC_TABLE_B[i] = tmp[1];
                }
            } else {
                EXP_FRAC_TABLE_A = FastMathLiteralArrays.loadExpFracA();
                EXP_FRAC_TABLE_B = FastMathLiteralArrays.loadExpFracB();
            }
        }
    }


    /**
     * Enclose large data table in nested static class so it's only loaded on first access.
     */
    private static class lnMant {

        /**
         * Extended precision logarithm table over the range 1 - 2 in increments of 2^-10.
         */
        private static final double[][] LN_MANT;

        static {
            if (RECOMPUTE_TABLES_AT_RUNTIME) {
                LN_MANT = new double[FastMath.LN_MANT_LEN][];

                // Populate lnMant table
                for (int i = 0; i < LN_MANT.length; i++) {
                    final double d = Double.longBitsToDouble(
                            (((long) i) << 42) | 0x3ff0000000000000L);

                    LN_MANT[i] = FastMathCalc.slowLog(d);
                }
            } else {
                LN_MANT = FastMathLiteralArrays.loadLnMant();
            }
        }
    }
}
