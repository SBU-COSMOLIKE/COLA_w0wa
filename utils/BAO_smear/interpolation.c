
#include "interpolation.h"
#include <gsl/gsl_interp.h>

int interpolate(double x[], double y[], double x_interp[], double y_interp[], int num_x, int num_x_interp, int type) {
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp *interp;

    if (type == 0)
        interp = gsl_interp_alloc(gsl_interp_linear, num_x);
    else
        interp = gsl_interp_alloc(gsl_interp_cspline, num_x);

    gsl_interp_init(interp, x, y, num_x);

    for (int i = 0; i < num_x_interp; i++) {
        y_interp[i] = gsl_interp_eval(interp, x, y, x_interp[i], acc);
    }

    gsl_interp_free(interp);
    gsl_interp_accel_free(acc);

    return 0;
}

