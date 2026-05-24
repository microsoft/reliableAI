package ch.idsia.blip.core.utils.other;


/**
 * Author Mihai Preda, 2006.
 * The author disclaims copyright to this source code.
 * <p/>
 * The method lgamma() is adapted from FDLIBM 5.3 (http://www.netlib.org/fdlibm/),
 * which comes with this copyright notice:
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * <p/>
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 * <p/>
 * The Lanczos and Stirling approximations are based on:
 * http://en.wikipedia.org/wiki/Lanczos_approximation
 * http://en.wikipedia.org/wiki/Stirling%27s_approximation
 * http://www.gnu.org/software/gsl/
 * http://jakarta.apache.org/commons/math/
 * http://my.fit.edu/~gabdo/gamma.txt
 */
@SuppressWarnings("JavaDoc")
public class Gamma {
    private static final double
            zero = 0.0;
    private static final double one = 1.0;
    private static final double two = 2.0;
    private static final double half = .5;
    private static final double SQRT2PI = 2.50662827463100024157;
    private static final double LN_SQRT2PI = 0.9189385332046727418;
    // coefficients for gamma=7, kmax=8  Lanczos method
    private static final double[] L9 = {
        0.99999999999980993227684700473478, 676.520368121885098567009190444019,
        -1259.13921672240287047156078755283, 771.3234287776530788486528258894,
        -176.61502916214059906584551354, 12.507343278686904814458936853,
        -0.13857109526572011689554707, 9.984369578019570859563e-6,
        1.50563273514931155834e-7
    };
    private static final double SQRT2PI_E7 = 0.0022857491179850424; // sqrt(2*pi)/e**7
    private static final double[] L15 = {
        0.99999999999999709182, 57.156235665862923517, -59.597960355475491248,
        14.136097974741747174, -0.49191381609762019978, .33994649984811888699e-4,
        .46523628927048575665e-4, -.98374475304879564677e-4,
        .15808870322491248884e-3, -.21026444172410488319e-3,
        .21743961811521264320e-3, -.16431810653676389022e-3,
        .84418223983852743293e-4, -.26190838401581408670e-4,
        .36899182659531622704e-5,
    };
    private static final double G_PLUS_HALF = (607 / 128.) + .5;
    private static final double
            SC1 = 0.08333333333333333;
    private static final double SC2 = 0.003472222222222222;
    private static final double SC3 = -0.0026813271604938273;
    private static final double SC4 = -2.2947209362139917E-4;
    private static final double LC1 = 0.08333333333333333;
    private static final double LC2 = -0.002777777777777778;
    private static final double LC3 = 7.936507936507937E-4;
    private static final double LC4 = -5.952380952380953E-4;
    private static final double
            a0 = 7.72156649015328655494e-02;
    private static final double a1 = 3.22467033424113591611e-01;
    private static final double a2 = 6.73523010531292681824e-02;
    private static final double a3 = 2.05808084325167332806e-02;
    private static final double a4 = 7.38555086081402883957e-03;
    private static final double a5 = 2.89051383673415629091e-03;
    private static final double a6 = 1.19270763183362067845e-03;
    private static final double a7 = 5.10069792153511336608e-04;
    private static final double a8 = 2.20862790713908385557e-04;
    private static final double a9 = 1.08011567247583939954e-04;
    private static final double a10 = 2.52144565451257326939e-05;
    private static final double a11 = 4.48640949618915160150e-05;
    private static final double tc = 1.46163214496836224576e+00;
    private static final double tf = -1.21486290535849611461e-01;
    private static final double tt = -3.63867699703950536541e-18;
    private static final double t0 = 4.83836122723810047042e-01;
    private static final double t1 = -1.47587722994593911752e-01;
    private static final double t2 = 6.46249402391333854778e-02;
    private static final double t3 = -3.27885410759859649565e-02;
    private static final double t4 = 1.79706750811820387126e-02;
    private static final double t5 = -1.03142241298341437450e-02;
    private static final double t6 = 6.10053870246291332635e-03;
    private static final double t7 = -3.68452016781138256760e-03;
    private static final double t8 = 2.25964780900612472250e-03;
    private static final double t9 = -1.40346469989232843813e-03;
    private static final double t10 = 8.81081882437654011382e-04;
    private static final double t11 = -5.38595305356740546715e-04;
    private static final double t12 = 3.15632070903625950361e-04;
    private static final double t13 = -3.12754168375120860518e-04;
    private static final double t14 = 3.35529192635519073543e-04;
    private static final double u0 = -7.72156649015328655494e-02;
    private static final double u1 = 6.32827064025093366517e-01;
    private static final double u2 = 1.45492250137234768737e+00;
    private static final double u3 = 9.77717527963372745603e-01;
    private static final double u4 = 2.28963728064692451092e-01;
    private static final double u5 = 1.33810918536787660377e-02;
    private static final double v1 = 2.45597793713041134822e+00;
    private static final double v2 = 2.12848976379893395361e+00;
    private static final double v3 = 7.69285150456672783825e-01;
    private static final double v4 = 1.04222645593369134254e-01;
    private static final double v5 = 3.21709242282423911810e-03;
    private static final double s0 = -7.72156649015328655494e-02;
    private static final double s1 = 2.14982415960608852501e-01;
    private static final double s2 = 3.25778796408930981787e-01;
    private static final double s3 = 1.46350472652464452805e-01;
    private static final double s4 = 2.66422703033638609560e-02;
    private static final double s5 = 1.84028451407337715652e-03;
    private static final double s6 = 3.19475326584100867617e-05;
    private static final double r1 = 1.39200533467621045958e+00;
    private static final double r2 = 7.21935547567138069525e-01;
    private static final double r3 = 1.71933865632803078993e-01;
    private static final double r4 = 1.86459191715652901344e-02;
    private static final double r5 = 7.77942496381893596434e-04;
    private static final double r6 = 7.32668430744625636189e-06;
    private static final double w0 = 4.18938533204672725052e-01;
    private static final double w1 = 8.33333333333329678849e-02;
    private static final double w2 = -2.77777777728775536470e-03;
    private static final double w3 = 7.93650558643019558500e-04;
    private static final double w4 = -5.95187557450339963135e-04;
    private static final double w5 = 8.36339918996282139126e-04;
    private static final double w6 = -1.63092934096575273989e-03;
    private static final double[] FACT = {
        1.0, 40320.0, 2.0922789888E13, 6.204484017332394E23,
        2.631308369336935E35, 8.159152832478977E47, 1.2413915592536073E61,
        7.109985878048635E74, 1.2688693218588417E89, 6.1234458376886085E103,
        7.156945704626381E118, 1.8548264225739844E134, 9.916779348709496E149,
        1.0299016745145628E166, 1.974506857221074E182, 6.689502913449127E198,
        3.856204823625804E215, 3.659042881952549E232, 5.5502938327393044E249,
        1.3113358856834524E267, 4.7147236359920616E284, 2.5260757449731984E302,
    };

    private static String ulps(double v, double ref) {
        double ulp = (ref == 0) ? Math.ulp(.1) : Math.ulp(ref);
        int ulps = (int) Math.floor(((v - ref) / ulp) + .5);

        // return ulps != 0 ? ""+ulps : "";
        return String.format("%d", ulps);
    }

    /**
     * Test the execution times and accuracy of different lgamma algorithms.
     *
     * @param argv not really used, here.
     */
    public static void main(String[] argv) {
        double a;
        double b = 0;
        double c = 0;
        double d = 0;
        double e = 0;

        for (int i = 1; i < 171; ++i) {
            a = Math.log(factorial(i));
            b = lgamma(i + 1);
            e = lanczosLGamma15(i);
            c = f(i);
            d = stirlingLGamma(i);

            System.out.printf("%3d | %6s | %6s | %6s | %6s |\n", i, ulps(b, a),
                    ulps(e, a), ulps(c, a), (i >= 10) ? ulps(d, a) : "-1000+");
            // System.out.printf("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n", thread, ulps(b, a), ulps(e, a), ulps(c, a), thread >= 10 ? ulps(d, a): "-1000+");
        }

        int N = 20000;
        long t1;
        long t2;

        t1 = System.currentTimeMillis();
        for (int r = 0; r < N; ++r) {
            for (int i = 1; i < 171; ++i) {
                b = lgamma(i + 1);
            }
        }
        t2 = System.currentTimeMillis();
        System.out.printf("fdlibm's 8: %d %f \n", t2 - t1, b);

        t1 = System.currentTimeMillis();
        for (int r = 0; r < N; ++r) {
            for (int i = 1; i < 171; ++i) {
                e = lanczosLGamma15(i);
            }
        }
        t2 = System.currentTimeMillis();
        System.out.printf("Lanczos 15 : %d %f \n", t2 - t1, e);

        t1 = System.currentTimeMillis();
        for (int r = 0; r < N; ++r) {
            for (int i = 1; i < 171; ++i) {
                c = f(i);
            }
        }
        t2 = System.currentTimeMillis();
        System.out.printf("f : %d %f \n", t2 - t1, c);

        t1 = System.currentTimeMillis();
        for (int r = 0; r < N; ++r) {
            for (int i = 1; i < 171; ++i) {
                c = lanczosLGamma9(i);
            }
        }
        t2 = System.currentTimeMillis();
        System.out.printf("Lanczos 8 : %d %f \n", t2 - t1, c);

        t1 = System.currentTimeMillis();
        for (int r = 0; r < N; ++r) {
            for (int i = 1; i < 171; ++i) {
                d = stirlingLGamma(i);
            }
        }
        t2 = System.currentTimeMillis();
        System.out.printf("Stirling  : %d %f \n", t2 - t1, d);
    }

    private static int HI(double x) {
        return (int) (Double.doubleToLongBits(x) >> 32);
    }

    private static int LO(double x) {
        return (int) Double.doubleToLongBits(x);
    }

    static double lanczosGamma9(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        double a = L9[0];

        for (int i = 1; i < 9; ++i) {
            a += L9[i] / (x + i);
        }
        return (SQRT2PI_E7 * a) * Math.pow((x + 7.5) / Math.E, x + .5);
    }

    private static double lanczosLGamma9(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        double a = L9[0];

        for (int i = 1; i < 9; ++i) {
            a += L9[i] / (x + i);
        }
        return ((LN_SQRT2PI + Math.log(a)) - 7.)
                + ((x + .5) * Math.log((x + 7.5) / Math.E));
    }

    private static double lanczosLGamma15(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        double a = L15[0];

        for (int i = 1; i < 15; ++i) {
            a += L15[i] / (x + i);
        }

        double tmp = x + G_PLUS_HALF;

        return ((LN_SQRT2PI + Math.log(a)) + ((x + .5) * Math.log(tmp))) - tmp;
    }

    static double g(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        double tmp = x + 5.2421875;

        return (0.9189385332046727418
                + Math.log(
                        0.99999999999999709182
                                + (57.156235665862923517 / (x + 1))
                                + (-59.597960355475491248 / (x + 2))
                                + (14.136097974741747174 / (x + 3))
                                + (-0.49191381609762019978 / (x + 4))
                                + (.33994649984811888699e-4 / (x + 5))
                                + (.46523628927048575665e-4 / (x + 6))
                                + (-.98374475304879564677e-4 / (x + 7))
                                + (.15808870322491248884e-3 / (x + 8))
                                + (-.21026444172410488319e-3 / (x + 9))
                                + (.21743961811521264320e-3 / (x + 10))
                                + (-.16431810653676389022e-3 / (x + 11))
                                + (.84418223983852743293e-4 / (x + 12))
                                + (-.26190838401581408670e-4 / (x + 13))
                                + (.36899182659531622704e-5 / (x + 14)))
                                + ((x + .5) * Math.log(tmp)))
                                        - tmp;
    }

    private static double f(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        double tmp = x + 5.2421875;

        // final double saveX = x;
        return (0.9189385332046727418
                + Math.log(
                        0.99999999999999709182 + (57.156235665862923517 / ++x)
                        + (-59.597960355475491248 / ++x)
                        + (14.136097974741747174 / ++x)
                        + (-0.49191381609762019978 / ++x)
                        + (.33994649984811888699e-4 / ++x)
                        + (.46523628927048575665e-4 / ++x)
                        + (-.98374475304879564677e-4 / ++x)
                        + (.15808870322491248884e-3 / ++x)
                        + (-.21026444172410488319e-3 / ++x)
                        + (.21743961811521264320e-3 / ++x)
                        + (-.16431810653676389022e-3 / ++x)
                        + (.84418223983852743293e-4 / ++x)
                        + (-.26190838401581408670e-4 / ++x)
                        + (.36899182659531622704e-5 / ++x))
                        + ((tmp - 4.7421875) * Math.log(tmp)))
                                - tmp// + (saveX + .5)*Math.log(tmp) + /*Math.sqrt(tmp)*/ - tmp
                                ;
    }

    static double stirlingGamma(double x) {
        double r1 = 1. / x;
        double r2 = r1 * r1;
        double r4 = r2 * r2;

        return SQRT2PI * Math.sqrt(x)
                * (1 + (SC1 * r1) + (SC2 * r2) + (SC3 * r1 * r2) + (SC4 * r4))
                * Math.pow(x / Math.E, x);
    }

    private static double stirlingLGamma(double x) {
        double r1 = 1. / x;
        double r2 = r1 * r1;
        double r3 = r1 * r2;
        double r5 = r2 * r3;
        double r7 = r3 * r3 * r1;

        return (((x + .5) * Math.log(x)) - x) + LN_SQRT2PI + (LC1 * r1)
                + (LC2 * r3) + (LC3 * r5) + (LC4 * r7);
    }

    private static double factorial(double x) {
        if (x <= -1) {
            return Double.NaN;
        }
        if (x <= 170) {
            if (Math.floor(x) == x) {
                int n = (int) x;
                double extra = x;

                switch (n & 7) {
                case 7:
                    extra *= --x;

                case 6:
                    extra *= --x;

                case 5:
                    extra *= --x;

                case 4:
                    extra *= --x;

                case 3:
                    extra *= --x;

                case 2:
                    extra *= --x;

                case 1:
                    return FACT[n >> 3] * extra;

                case 0:
                    return FACT[n >> 3];
                }
            }
        }
        return Math.exp(lgamma(x + 1));
    }

    /**
     * @param b input value
     * @return log gamma of given value.
     */
    public static double lgamma(double b) {
        double t;
        double a;
        double z;
        double p;
        double p1;
        double p2;
        double p3;
        double q;
        double r;
        double w;
        int i;

        int hx = HI(b);
        int lx = LO(b);

        /* purge off +-inf, NaN, +-0, and negative arguments */
        int ix = hx & 0x7fffffff;

        if (ix >= 0x7ff00000) {
            return Double.POSITIVE_INFINITY;
        }
        if (((ix | lx) == 0) || (hx < 0)) {
            return Double.NaN;
        }
        if (ix < 0x3b900000) { /* |x|<2**-70, return -log(|x|) */
            return -Math.log(b);
        }

        /* purge off 1 and 2 */
        if ((((ix - 0x3ff00000) | lx) == 0) || (((ix - 0x40000000) | lx) == 0)) {
            r = 0;
        } /* for x < 2.0 */else if (ix < 0x40000000) {
            if (ix <= 0x3feccccc) { /* lgamma(x) = lgamma(x+1)-log(x) */
                r = -Math.log(b);
                if (ix >= 0x3FE76944) {
                    a = one - b;
                    i = 0;
                } else if (ix >= 0x3FCDA661) {
                    a = b - (tc - one);
                    i = 1;
                } else {
                    a = b;
                    i = 2;
                }
            } else {
                r = zero;
                if (ix >= 0x3FFBB4C3) {
                    a = 2.0 - b;
                    i = 0;
                } /* [1.7316,2] */else if (ix >= 0x3FF3B4C4) {
                    a = b - tc;
                    i = 1;
                } /* [1.23,1.73] */else {
                    a = b - one;
                    i = 2;
                }
            }

            switch (i) {
            case 0:
                z = a * a;
                p1 = a0
                        + z
                                * (a2
                                        + (z
                                                * (a4
                                                        + (z
                                                                * (a6
                                                                        + (z
                                                                                * (a8
                                                                                        + (z
                                                                                                * a10))))))));
                p2 = z
                        * (a1
                                + (z
                                        * (a3
                                                + (z
                                                        * (a5
                                                                + (z
                                                                        * (a7
                                                                                + (z
                                                                                        * (a9
                                                                                                + (z
                                                                                                        * a11))))))))));
                p = a * p1 + p2;
                r += (p - (0.5 * a));
                break;

            case 1:
                z = a * a;
                w = z * a;
                p1 = t0 + w * (t3 + (w * (t6 + (w * (t9 + (w * t12)))))); /* parallel comp */
                p2 = t1
                        + w * (t4 + (w * (t7 + (w * (t10 + (w * t13))))));
                p3 = t2 + w * (t5 + (w * (t8 + (w * (t11 + (w * t14))))));
                p = z * p1 - (tt - (w * (p2 + (a * p3))));
                r += (tf + p);
                break;

            case 2:
                p1 = a
                        * (u0
                                + (a
                                        * (u1
                                                + (a
                                                        * (u2
                                                                + (a
                                                                        * (u3
                                                                                + (a
                                                                                        * (u4
                                                                                                + (a
                                                                                                        * u5))))))))));
                p2 = one
                        + a
                                * (v1
                                        + (a
                                                * (v2
                                                        + (a
                                                                * (v3
                                                                        + (a
                                                                                * (v4
                                                                                        + (a
                                                                                                * v5))))))));
                r += ((-0.5 * a) + (p1 / p2));
            }
        } else if (ix < 0x40200000) { /* x < 8.0 */
            i = (int) b;
            // t = zero;
            a = b - (double) i;
            p = a
                    * (s0
                            + (a
                                    * (s1
                                            + (a
                                                    * (s2
                                                            + (a
                                                                    * (s3
                                                                            + (a
                                                                                    * (s4
                                                                                            + (a
                                                                                                    * (s5
                                                                                                            + (a
                                                                                                                    * s6))))))))))));
            q = one
                    + a
                            * (r1
                                    + (a
                                            * (r2
                                                    + (a
                                                            * (r3
                                                                    + (a
                                                                            * (r4
                                                                                    + (a
                                                                                            * (r5
                                                                                                    + (a
                                                                                                            * r6))))))))));
            r = half * a + p / q;
            z = one; /* lgamma(1+s) = log(s) + lgamma(s) */
            switch (i) {
            case 7:
                z *= (a + 6.0); /* FALLTHRU */

            case 6:
                z *= (a + 5.0); /* FALLTHRU */

            case 5:
                z *= (a + 4.0); /* FALLTHRU */

            case 4:
                z *= (a + 3.0); /* FALLTHRU */

            case 3:
                z *= (a + 2.0); /* FALLTHRU */
                r += Math.log(z);
                break;
            }

            /* 8.0 <= x < 2**58 */
        } else if (ix < 0x43900000) {
            t = Math.log(b);
            z = one / b;
            a = z * z;
            w = w0
                    + z
                            * (w1
                                    + (a
                                            * (w2
                                                    + (a
                                                            * (w3
                                                                    + (a
                                                                            * (w4
                                                                                    + (a
                                                                                            * (w5
                                                                                                    + (a
                                                                                                            * w6))))))))));
            r = (b - half) * (t - one) + w;
        } else {

            /* 2**58 <= x <= inf */
            r = b * (Math.log(b) - one);
        }
        return r;
    }
}
