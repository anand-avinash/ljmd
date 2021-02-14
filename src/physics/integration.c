#include "physics.h"

// split the verlet algorithm into two functions

/* first part: propagate velocities by half and positions by full step  */
void verlet_1(mdsys_t *sys) {

        #ifdef _OMP_NAIVE
        #pragma omp parallel for default(shared)
        #endif
        for (int i = 0; i < sys->natoms; ++i) {
                sys->v[i].x += 0.5 * sys->dt / mvsq2e * sys->f[i].x / sys->mass;
                sys->v[i].y += 0.5 * sys->dt / mvsq2e * sys->f[i].y / sys->mass;
                sys->v[i].z += 0.5 * sys->dt / mvsq2e * sys->f[i].z / sys->mass;
                sys->r[i].x += sys->dt * sys->v[i].x;
                sys->r[i].y += sys->dt * sys->v[i].y;
                sys->r[i].z += sys->dt * sys->v[i].z;
        }
}

/* second part: propagate velocities by another half step */
void verlet_2(mdsys_t *sys) {

        #ifdef _OMP_NAIVE
        #pragma omp parallel for default(shared)
        #endif
        for (int i = 0; i < sys->natoms; ++i) {
                sys->v[i].x += 0.5 * sys->dt / mvsq2e * sys->f[i].x / sys->mass;
                sys->v[i].y += 0.5 * sys->dt / mvsq2e * sys->f[i].y / sys->mass;
                sys->v[i].z += 0.5 * sys->dt / mvsq2e * sys->f[i].z / sys->mass;
        }
}
